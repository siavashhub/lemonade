#include "lemon/mcp_server.h"

#include <exception>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <lemon/utils/aixlog.hpp>

#include "lemon/utils/json_utils.h"
#include "lemon/version.h"

namespace lemon {

namespace {

// JSON-RPC 2.0 error codes (https://www.jsonrpc.org/specification#error_object).
constexpr int kJsonRpcParseError      = -32700;
constexpr int kJsonRpcInvalidRequest  = -32600;
constexpr int kJsonRpcMethodNotFound  = -32601;
constexpr int kJsonRpcInvalidParams   = -32602;
constexpr int kJsonRpcInternalError   = -32603;

// MCP protocol version implemented by this server (the "Streamable HTTP"
// transport, JSON-RPC 2.0 framing, tools capability only).
constexpr const char* kMcpProtocolVersion = "2025-06-18";

// Identity reported in the `initialize` response's `serverInfo`.
constexpr const char* kServerName = "lemonade-mcp";

bool is_notification(const json& message) {
    // A JSON-RPC message is a notification if it has no `id` field. (A null
    // id is still a valid request, but the spec discourages it.)
    return !message.contains("id");
}

std::string extract_string_arg(const json& args, const char* key) {
    if (!args.contains(key) || !args[key].is_string()) {
        throw std::runtime_error(std::string("Missing or non-string argument: ") + key);
    }
    return args[key].get<std::string>();
}

}  // namespace

McpServer::McpServer(Router* router, ModelManager* model_manager, EnsureLoadedFn ensure_loaded)
    : router_(router),
      model_manager_(model_manager),
      ensure_loaded_(std::move(ensure_loaded)) {}

// =============================================================================
// HTTP route registration
// =============================================================================

void McpServer::register_routes(httplib::Server& server) {
    auto self = shared_from_this();

    server.Post("/mcp", [self](const httplib::Request& req, httplib::Response& res) {
        std::string response_body;
        try {
            response_body = self->handle_request_body(req.body);
        } catch (const std::exception& e) {
            LOG(ERROR, "McpServer") << "Unhandled exception in POST /mcp: " << e.what() << std::endl;
            res.status = 500;
            json err = make_error_response(nullptr, kJsonRpcInternalError, e.what());
            res.set_content(err.dump(), "application/json");
            return;
        }

        if (response_body.empty()) {
            // All messages in the batch were notifications — JSON-RPC says no
            // response body should be returned. MCP's Streamable HTTP transport
            // expects 202 Accepted in that case.
            res.status = 202;
            return;
        }

        res.status = 200;
        res.set_content(response_body, "application/json");
    });

    server.Get("/mcp", [](const httplib::Request&, httplib::Response& res) {
        // The Streamable HTTP transport allows GET for opening an SSE channel
        // for server-initiated messages. This MVP gateway is request/response
        // only, so we explicitly refuse GET to make the limitation visible.
        res.status = 405;
        res.set_header("Allow", "POST");
        res.set_content(
            "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32600,"
            "\"message\":\"GET /mcp not supported; use POST with a JSON-RPC body\"},"
            "\"id\":null}",
            "application/json");
    });
}

// =============================================================================
// Top-level JSON-RPC envelope handling
// =============================================================================

std::string McpServer::handle_request_body(const std::string& body) {
    json parsed;
    try {
        parsed = json::parse(body);
    } catch (const json::parse_error& e) {
        json err = make_error_response(nullptr, kJsonRpcParseError,
                                       std::string("Invalid JSON: ") + e.what());
        return err.dump();
    }

    // Batch request: array of messages.
    if (parsed.is_array()) {
        if (parsed.empty()) {
            json err = make_error_response(nullptr, kJsonRpcInvalidRequest,
                                           "Batch request must not be empty");
            return err.dump();
        }
        json responses = json::array();
        for (const auto& message : parsed) {
            json reply = dispatch_message(message);
            if (!reply.is_null()) {
                responses.push_back(std::move(reply));
            }
        }
        if (responses.empty()) {
            return "";  // All notifications.
        }
        return responses.dump();
    }

    // Single message.
    if (!parsed.is_object()) {
        json err = make_error_response(nullptr, kJsonRpcInvalidRequest,
                                       "Request must be a JSON object or array");
        return err.dump();
    }
    json reply = dispatch_message(parsed);
    if (reply.is_null()) {
        return "";
    }
    return reply.dump();
}

json McpServer::dispatch_message(const json& message) {
    // Validate envelope shape.
    if (!message.is_object()) {
        return make_error_response(nullptr, kJsonRpcInvalidRequest,
                                   "Request must be a JSON object");
    }

    const bool notification = is_notification(message);
    json id = message.value("id", json(nullptr));

    if (!message.contains("method") || !message["method"].is_string()) {
        if (notification) return json(nullptr);
        return make_error_response(id, kJsonRpcInvalidRequest, "Missing `method` field");
    }

    const std::string method = message["method"].get<std::string>();
    json params = message.value("params", json::object());

    try {
        if (method == "initialize") {
            if (notification) return json(nullptr);
            return handle_initialize(params, id);
        }
        if (method == "tools/list") {
            if (notification) return json(nullptr);
            return handle_tools_list(id);
        }
        if (method == "tools/call") {
            if (notification) return json(nullptr);
            return handle_tools_call(params, id);
        }
        if (method == "ping") {
            if (notification) return json(nullptr);
            return handle_ping(id);
        }
        // `notifications/initialized` and `notifications/cancelled` arrive as
        // notifications. We acknowledge them silently — no state to update.
        if (method == "notifications/initialized" ||
            method == "notifications/cancelled") {
            return json(nullptr);
        }

        if (notification) return json(nullptr);
        return make_error_response(id, kJsonRpcMethodNotFound,
                                   "Unknown method: " + method);
    } catch (const std::exception& e) {
        LOG(ERROR, "McpServer") << "Method '" << method << "' threw: " << e.what() << std::endl;
        if (notification) return json(nullptr);
        return make_error_response(id, kJsonRpcInternalError, e.what());
    }
}

// =============================================================================
// MCP method handlers
// =============================================================================

json McpServer::handle_initialize(const json& /*params*/, const json& id) {
    json result = {
        {"protocolVersion", kMcpProtocolVersion},
        {"capabilities", {
            {"tools", json::object()},  // Tools capability with no extra options.
        }},
        {"serverInfo", {
            {"name", kServerName},
            {"version", LEMON_VERSION_STRING},
        }},
    };
    return make_success_response(id, std::move(result));
}

json McpServer::handle_tools_list(const json& id) {
    json result = {{"tools", tools_descriptor()}};
    return make_success_response(id, std::move(result));
}

json McpServer::handle_tools_call(const json& params, const json& id) {
    if (!params.is_object() || !params.contains("name") || !params["name"].is_string()) {
        return make_error_response(id, kJsonRpcInvalidParams,
                                   "tools/call requires a string `name` parameter");
    }
    const std::string tool_name = params["name"].get<std::string>();
    json arguments = params.value("arguments", json::object());
    if (!arguments.is_object()) {
        return make_error_response(id, kJsonRpcInvalidParams,
                                   "tools/call `arguments` must be an object");
    }

    LOG(INFO, "McpServer") << "tools/call: " << tool_name << std::endl;

    json result;
    try {
        if (tool_name == "lemonade_chat") {
            result = tool_chat(arguments);
        } else if (tool_name == "lemonade_embed") {
            result = tool_embed(arguments);
        } else if (tool_name == "lemonade_transcribe_audio") {
            result = tool_transcribe_audio(arguments);
        } else if (tool_name == "lemonade_generate_speech") {
            result = tool_generate_speech(arguments);
        } else if (tool_name == "lemonade_generate_image") {
            result = tool_generate_image(arguments);
        } else {
            // Per MCP spec, unknown-tool errors are returned as a successful
            // result with isError=true (so the client model can recover),
            // *not* as a JSON-RPC method-not-found error.
            result = {
                {"content", json::array({text_content_block("Unknown tool: " + tool_name)})},
                {"isError", true},
            };
        }
    } catch (const std::exception& e) {
        LOG(ERROR, "McpServer") << "Tool '" << tool_name << "' failed: " << e.what() << std::endl;
        result = {
            {"content", json::array({text_content_block(std::string("Error: ") + e.what())})},
            {"isError", true},
        };
    }

    return make_success_response(id, std::move(result));
}

json McpServer::handle_ping(const json& id) {
    return make_success_response(id, json::object());
}

// =============================================================================
// Tool implementations
// =============================================================================

json McpServer::tool_chat(const json& arguments) {
    const std::string model = extract_string_arg(arguments, "model");
    if (!arguments.contains("messages") || !arguments["messages"].is_array()) {
        throw std::runtime_error("Missing or non-array argument: messages");
    }

    ensure_loaded_(model);

    // Translate MCP tool args → OpenAI chat completion request. We
    // deliberately do NOT enable streaming: MCP tool calls are
    // request/response, and any streaming output would be wasted.
    json openai_request = {
        {"model", model},
        {"messages", arguments["messages"]},
        {"stream", false},
    };

    // Forward common optional knobs verbatim.
    for (const char* key : {"temperature", "top_p", "max_tokens", "stop",
                            "seed", "presence_penalty", "frequency_penalty",
                            "tools", "tool_choice", "response_format"}) {
        if (arguments.contains(key)) {
            openai_request[key] = arguments[key];
        }
    }

    json response = router_->chat_completion(openai_request);
    if (response.contains("error")) {
        throw std::runtime_error(response["error"].value("message", "chat completion failed"));
    }

    // Surface assistant content as a text block. If the model emitted tool
    // calls, attach them as a second text block (JSON-stringified) so MCP
    // clients can see them — MCP doesn't have a first-class tool_call content
    // type, so embedding as JSON text is the standard convention.
    std::string assistant_text;
    json tool_calls;
    if (response.contains("choices") && response["choices"].is_array() &&
        !response["choices"].empty()) {
        const auto& message = response["choices"][0].value("message", json::object());
        if (message.contains("content") && message["content"].is_string()) {
            assistant_text = message["content"].get<std::string>();
        }
        if (message.contains("tool_calls")) {
            tool_calls = message["tool_calls"];
        }
    }

    json content = json::array();
    content.push_back(text_content_block(assistant_text));
    if (!tool_calls.is_null() && !tool_calls.empty()) {
        content.push_back(text_content_block(
            std::string("tool_calls: ") + tool_calls.dump()));
    }

    return json{
        {"content", std::move(content)},
        {"isError", false},
    };
}

json McpServer::tool_embed(const json& arguments) {
    const std::string model = extract_string_arg(arguments, "model");
    if (!arguments.contains("input")) {
        throw std::runtime_error("Missing argument: input");
    }

    ensure_loaded_(model);

    json openai_request = {
        {"model", model},
        {"input", arguments["input"]},
    };
    if (arguments.contains("encoding_format")) {
        openai_request["encoding_format"] = arguments["encoding_format"];
    }

    json response = router_->embeddings(openai_request);
    if (response.contains("error")) {
        throw std::runtime_error(response["error"].value("message", "embeddings failed"));
    }

    // MCP has no embedding content type, so we return the embedding vectors
    // as a JSON-stringified text block. Callers parse the text as JSON to
    // get the array of vectors.
    json out = json::object();
    out["model"] = response.value("model", model);
    out["data"] = response.value("data", json::array());
    if (response.contains("usage")) {
        out["usage"] = response["usage"];
    }

    return json{
        {"content", json::array({text_content_block(out.dump())})},
        {"isError", false},
    };
}

json McpServer::tool_transcribe_audio(const json& arguments) {
    const std::string model = extract_string_arg(arguments, "model");
    const std::string audio_b64 = extract_string_arg(arguments, "audio_base64");
    const std::string filename = arguments.value("filename", std::string("audio.wav"));

    ensure_loaded_(model);

    // The router's audio_transcriptions expects raw bytes in `file_data`
    // (matching how handle_audio_transcriptions stuffs the multipart upload).
    json router_request = {
        {"model", model},
        {"file_data", utils::JsonUtils::base64_decode(audio_b64)},
        {"filename", filename},
    };
    for (const char* key : {"language", "prompt", "response_format", "temperature"}) {
        if (arguments.contains(key)) {
            router_request[key] = arguments[key];
        }
    }

    json response = router_->audio_transcriptions(router_request);
    if (response.contains("error")) {
        throw std::runtime_error(response["error"].value("message", "transcription failed"));
    }

    // The OpenAI-shaped response carries the transcript under `text`. Surface
    // it as a text block; attach the full structured response as JSON for
    // callers that need timestamps / segments.
    std::string transcript = response.value("text", std::string());
    json content = json::array();
    content.push_back(text_content_block(transcript));
    content.push_back(text_content_block(response.dump()));

    return json{
        {"content", std::move(content)},
        {"isError", false},
    };
}

json McpServer::tool_generate_speech(const json& arguments) {
    const std::string model = extract_string_arg(arguments, "model");
    const std::string input = extract_string_arg(arguments, "input");
    const std::string voice = arguments.value("voice", std::string("af_heart"));
    const std::string response_format = arguments.value("response_format", std::string("mp3"));

    ensure_loaded_(model);

    json router_request = {
        {"model", model},
        {"input", input},
        {"voice", voice},
        {"response_format", response_format},
    };

    // Capture the streamed audio into an in-memory buffer. All three callbacks
    // must be wired — the streaming proxy calls done() at end-of-stream, and
    // a default-constructed std::function would throw bad_function_call.
    // (Pattern mirrors collection_orchestrator.cpp.)
    std::string buffer;
    httplib::DataSink sink;
    sink.write = [&buffer](const char* data, size_t len) {
        buffer.append(data, len);
        return true;
    };
    sink.is_writable = []() { return true; };
    sink.done = []() {};
    router_->audio_speech(router_request, sink);

    if (buffer.empty()) {
        throw std::runtime_error("Text-to-speech produced no audio");
    }

    // MCP's audio content block: { type: "audio", data: <base64>, mimeType: <string> }.
    static const std::unordered_map<std::string, std::string> kMimeTypes = {
        {"mp3", "audio/mpeg"},
        {"wav", "audio/wav"},
        {"opus", "audio/opus"},
        {"flac", "audio/flac"},
        {"pcm", "audio/pcm"},
        {"aac", "audio/aac"},
    };
    auto mime_it = kMimeTypes.find(response_format);
    const std::string mime_type = (mime_it != kMimeTypes.end()) ? mime_it->second : "audio/mpeg";

    json audio_block = {
        {"type", "audio"},
        {"data", utils::JsonUtils::base64_encode(buffer)},
        {"mimeType", mime_type},
    };

    return json{
        {"content", json::array({std::move(audio_block)})},
        {"isError", false},
    };
}

json McpServer::tool_generate_image(const json& arguments) {
    const std::string model = extract_string_arg(arguments, "model");
    const std::string prompt = extract_string_arg(arguments, "prompt");

    ensure_loaded_(model);

    json router_request = {
        {"model", model},
        {"prompt", prompt},
    };
    for (const char* key : {"size", "n", "quality", "style", "response_format",
                            "negative_prompt", "seed", "steps", "cfg_scale"}) {
        if (arguments.contains(key)) {
            router_request[key] = arguments[key];
        }
    }
    // Force base64 output — MCP image content blocks carry inline base64,
    // not URLs, and Lemonade's image backends already default to b64_json.
    router_request["response_format"] = "b64_json";

    json response = router_->image_generations(router_request);
    if (response.contains("error")) {
        throw std::runtime_error(response["error"].value("message", "image generation failed"));
    }

    json content = json::array();
    if (response.contains("data") && response["data"].is_array()) {
        for (const auto& entry : response["data"]) {
            if (!entry.contains("b64_json") || !entry["b64_json"].is_string()) continue;
            content.push_back({
                {"type", "image"},
                {"data", entry["b64_json"]},
                {"mimeType", "image/png"},
            });
        }
    }
    if (content.empty()) {
        throw std::runtime_error("Image generation returned no images");
    }

    return json{
        {"content", std::move(content)},
        {"isError", false},
    };
}

// =============================================================================
// Tool catalogue (consumed by tools/list)
// =============================================================================

json McpServer::tools_descriptor() {
    return json::array({
        {
            {"name", "lemonade_chat"},
            {"description",
             "Chat completion against a locally hosted LLM. Pass a `messages` "
             "array (OpenAI chat format) and the model name. Returns the "
             "assistant text response."},
            {"inputSchema", {
                {"type", "object"},
                {"required", json::array({"model", "messages"})},
                {"properties", {
                    {"model",       {{"type", "string"},
                                     {"description", "Lemonade model name (e.g. Qwen3-0.6B-GGUF)"}}},
                    {"messages",    {{"type", "array"},
                                     {"description", "OpenAI-format chat messages"},
                                     {"items", {{"type", "object"}}}}},
                    {"temperature", {{"type", "number"}}},
                    {"top_p",       {{"type", "number"}}},
                    {"max_tokens",  {{"type", "integer"}}},
                    {"stop",        {{"description", "stop sequences (string or array)"}}},
                    {"seed",        {{"type", "integer"}}},
                    {"tools",       {{"type", "array"},
                                     {"description", "OpenAI-format tool definitions"},
                                     {"items", {{"type", "object"}}}}},
                    {"tool_choice", {{"description", "auto | none | required | {type: function, ...}"}}},
                    {"response_format", {{"type", "object"}}},
                }},
            }},
        },
        {
            {"name", "lemonade_embed"},
            {"description",
             "Generate an embedding vector for one or more input strings using "
             "a locally hosted embedding model. Returns a JSON-encoded text "
             "block with `{model, data: [{embedding, index, object}], usage}`."},
            {"inputSchema", {
                {"type", "object"},
                {"required", json::array({"model", "input"})},
                {"properties", {
                    {"model", {{"type", "string"}}},
                    {"input", {{"description", "string OR array of strings"}}},
                    {"encoding_format", {{"type", "string"},
                                         {"enum", json::array({"float", "base64"})}}},
                }},
            }},
        },
        {
            {"name", "lemonade_transcribe_audio"},
            {"description",
             "Transcribe an audio clip with a Whisper-class model. Audio must "
             "be base64-encoded. Returns the transcript as the first text "
             "block, followed by the full OpenAI-shaped JSON response."},
            {"inputSchema", {
                {"type", "object"},
                {"required", json::array({"model", "audio_base64"})},
                {"properties", {
                    {"model",         {{"type", "string"}}},
                    {"audio_base64",  {{"type", "string"},
                                       {"description", "Audio file bytes, base64-encoded"}}},
                    {"filename",      {{"type", "string"},
                                       {"description", "Hint for content type (e.g. \"clip.wav\")"}}},
                    {"language",      {{"type", "string"}}},
                    {"prompt",        {{"type", "string"}}},
                    {"response_format", {{"type", "string"},
                                         {"enum", json::array({"json", "text", "srt", "verbose_json", "vtt"})}}},
                    {"temperature",   {{"type", "number"}}},
                }},
            }},
        },
        {
            {"name", "lemonade_generate_speech"},
            {"description",
             "Synthesize speech from text with a locally hosted TTS model. "
             "Returns a single audio content block with base64-encoded audio."},
            {"inputSchema", {
                {"type", "object"},
                {"required", json::array({"model", "input"})},
                {"properties", {
                    {"model",  {{"type", "string"}}},
                    {"input",  {{"type", "string"}, {"description", "Text to speak"}}},
                    {"voice",  {{"type", "string"}, {"description", "Voice name (e.g. af_heart)"}}},
                    {"response_format", {{"type", "string"},
                                         {"enum", json::array({"mp3", "wav", "opus", "flac", "pcm", "aac"})}}},
                }},
            }},
        },
        {
            {"name", "lemonade_generate_image"},
            {"description",
             "Generate one or more images from a text prompt with a locally "
             "hosted image model. Always returns base64-encoded PNGs."},
            {"inputSchema", {
                {"type", "object"},
                {"required", json::array({"model", "prompt"})},
                {"properties", {
                    {"model",  {{"type", "string"}}},
                    {"prompt", {{"type", "string"}}},
                    {"size",   {{"type", "string"},
                                {"description", "e.g. 512x512, 1024x1024"}}},
                    {"n",      {{"type", "integer"}, {"minimum", 1}}},
                    {"negative_prompt", {{"type", "string"}}},
                    {"seed",   {{"type", "integer"}}},
                    {"steps",  {{"type", "integer"}}},
                    {"cfg_scale", {{"type", "number"}}},
                }},
            }},
        },
    });
}

// =============================================================================
// JSON-RPC envelope helpers
// =============================================================================

json McpServer::make_error_response(const json& id, int code, const std::string& message) {
    return {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"error", {
            {"code", code},
            {"message", message},
        }},
    };
}

json McpServer::make_success_response(const json& id, json result) {
    return {
        {"jsonrpc", "2.0"},
        {"id", id},
        {"result", std::move(result)},
    };
}

json McpServer::text_content_block(const std::string& text) {
    return {
        {"type", "text"},
        {"text", text},
    };
}

}  // namespace lemon
