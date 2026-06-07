#include "lemon/mcp_server.h"

#include <exception>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <lemon/utils/aixlog.hpp>

#include "lemon/utils/json_utils.h"
#include "lemon/version.h"

namespace lemon {

namespace {

constexpr int kJsonRpcParseError      = -32700;
constexpr int kJsonRpcInvalidRequest  = -32600;
constexpr int kJsonRpcMethodNotFound  = -32601;
constexpr int kJsonRpcInvalidParams   = -32602;
constexpr int kJsonRpcInternalError   = -32603;

constexpr const char* kMcpProtocolVersion = "2025-06-18";
constexpr const char* kServerName = "lemonade-mcp";

bool is_notification(const json& message) {
    return !message.contains("id");
}

std::string extract_string_arg(const json& args, const char* key) {
    if (!args.contains(key) || !args[key].is_string()) {
        throw std::runtime_error(std::string("Missing or non-string argument: ") + key);
    }
    return args[key].get<std::string>();
}

// Some MCP clients (notably Claude Desktop) emit chat messages in Anthropic's
// content-block form — `content: [{type:"text", text:"..."}]` — instead of
// OpenAI's plain-string form. llama.cpp silently returns empty content when it
// sees the array form. Flatten any text blocks back into a single string.
json normalize_messages(const json& messages) {
    if (!messages.is_array()) return messages;
    json out = json::array();
    for (const auto& msg : messages) {
        if (!msg.is_object() || !msg.contains("content") || !msg["content"].is_array()) {
            out.push_back(msg);
            continue;
        }
        json normalized = msg;
        std::string text;
        for (const auto& block : msg["content"]) {
            if (block.is_string()) {
                text += block.get<std::string>();
            } else if (block.is_object() && block.value("type", std::string()) == "text" &&
                       block.contains("text") && block["text"].is_string()) {
                text += block["text"].get<std::string>();
            }
        }
        normalized["content"] = text;
        out.push_back(std::move(normalized));
    }
    return out;
}

}  // namespace

McpServer::McpServer(Router* router, ModelManager* model_manager, EnsureLoadedFn ensure_loaded)
    : router_(router),
      model_manager_(model_manager),
      ensure_loaded_(std::move(ensure_loaded)) {}

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
            // All messages were notifications — Streamable HTTP expects 202.
            res.status = 202;
            return;
        }

        res.status = 200;
        res.set_content(response_body, "application/json");
    });

    server.Get("/mcp", [](const httplib::Request&, httplib::Response& res) {
        // No SSE channel in this MVP; refuse GET explicitly.
        res.status = 405;
        res.set_header("Allow", "POST");
        res.set_content(
            "{\"jsonrpc\":\"2.0\",\"error\":{\"code\":-32600,"
            "\"message\":\"GET /mcp not supported; use POST with a JSON-RPC body\"},"
            "\"id\":null}",
            "application/json");
    });
}

std::string McpServer::handle_request_body(const std::string& body) {
    json parsed;
    try {
        parsed = json::parse(body);
    } catch (const json::parse_error& e) {
        json err = make_error_response(nullptr, kJsonRpcParseError,
                                       std::string("Invalid JSON: ") + e.what());
        return err.dump();
    }

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
        if (responses.empty()) return "";
        return responses.dump();
    }

    if (!parsed.is_object()) {
        json err = make_error_response(nullptr, kJsonRpcInvalidRequest,
                                       "Request must be a JSON object or array");
        return err.dump();
    }
    json reply = dispatch_message(parsed);
    if (reply.is_null()) return "";
    return reply.dump();
}

json McpServer::dispatch_message(const json& message) {
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

json McpServer::handle_initialize(const json& /*params*/, const json& id) {
    json result = {
        {"protocolVersion", kMcpProtocolVersion},
        {"capabilities", {
            {"tools", json::object()},
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
        } else if (tool_name == "lemonade_transcribe_audio") {
            result = tool_transcribe_audio(arguments);
        } else if (tool_name == "lemonade_generate_image") {
            result = tool_generate_image(arguments);
        } else if (tool_name == "lemonade_list_models") {
            result = tool_list_models(arguments);
        } else {
            // Per MCP spec, unknown-tool errors are isError=true results, not JSON-RPC errors.
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

json McpServer::tool_chat(const json& arguments) {
    const std::string model = extract_string_arg(arguments, "model");
    if (!arguments.contains("messages") || !arguments["messages"].is_array()) {
        throw std::runtime_error("Missing or non-array argument: messages");
    }

    ensure_loaded_(model);

    json openai_request = {
        {"model", model},
        {"messages", normalize_messages(arguments["messages"])},
        {"stream", false},
    };

    for (const char* key : {"temperature", "top_p", "max_tokens", "stop",
                            "seed", "presence_penalty", "frequency_penalty",
                            "tools", "tool_choice", "response_format",
                            "chat_template_kwargs"}) {
        if (arguments.contains(key)) {
            openai_request[key] = arguments[key];
        }
    }

    // Reasoning models (Qwen3, DeepSeek-R1) burn small max_tokens budgets on
    // <think> blocks, leaving content empty. Disable thinking by default;
    // callers can opt back in via chat_template_kwargs.enable_thinking=true.
    if (!openai_request.contains("chat_template_kwargs")) {
        openai_request["chat_template_kwargs"] = {{"enable_thinking", false}};
    }

    json response = router_->chat_completion(openai_request);
    if (response.contains("error")) {
        throw std::runtime_error(response["error"].value("message", "chat completion failed"));
    }

    // Reasoning-model fallback: if `content` is empty but `reasoning_content`
    // is not, surface the reasoning rather than returning an empty string.
    std::string assistant_text;
    std::string reasoning_text;
    std::string finish_reason;
    json tool_calls;
    if (response.contains("choices") && response["choices"].is_array() &&
        !response["choices"].empty()) {
        const auto& choice = response["choices"][0];
        finish_reason = choice.value("finish_reason", std::string());
        const auto& message = choice.value("message", json::object());
        if (message.contains("content") && message["content"].is_string()) {
            assistant_text = message["content"].get<std::string>();
        }
        if (message.contains("reasoning_content") &&
            message["reasoning_content"].is_string()) {
            reasoning_text = message["reasoning_content"].get<std::string>();
        }
        if (message.contains("tool_calls")) {
            tool_calls = message["tool_calls"];
        }
    }

    json content = json::array();
    if (assistant_text.empty() && !reasoning_text.empty()) {
        std::string note = "[reasoning only";
        if (finish_reason == "length") note += "; finish_reason=length";
        note += "]\n";
        content.push_back(text_content_block(note + reasoning_text));
    } else {
        content.push_back(text_content_block(assistant_text));
    }
    if (!tool_calls.is_null() && !tool_calls.empty()) {
        content.push_back(text_content_block(
            std::string("tool_calls: ") + tool_calls.dump()));
    }

    return json{
        {"content", std::move(content)},
        {"isError", false},
    };
}

json McpServer::tool_transcribe_audio(const json& arguments) {
    const std::string model = extract_string_arg(arguments, "model");

    // Same-machine deployment: prefer `audio_path` over base64 to avoid
    // multi-MB blobs through JSON-RPC. Agents reach for `audio_path` first;
    // accepting only base64 led to path-in-base64 errors.
    std::string audio_data;
    std::string filename = arguments.value("filename", std::string());
    const bool has_path = arguments.contains("audio_path") && arguments["audio_path"].is_string();
    const bool has_b64  = arguments.contains("audio_base64") && arguments["audio_base64"].is_string();
    if (!has_path && !has_b64) {
        throw std::runtime_error(
            "Provide either `audio_path` (path to a local audio file) or "
            "`audio_base64` (base64-encoded audio bytes).");
    }
    if (has_path) {
        const std::string path = arguments["audio_path"].get<std::string>();
        std::error_code ec;
        if (!std::filesystem::exists(path, ec) || ec) {
            throw std::runtime_error("audio_path not found: " + path);
        }
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error("Failed to open audio_path: " + path);
        }
        std::ostringstream buf;
        buf << in.rdbuf();
        audio_data = buf.str();
        if (audio_data.empty()) {
            throw std::runtime_error("audio_path is empty: " + path);
        }
        if (filename.empty()) {
            filename = std::filesystem::path(path).filename().string();
        }
    } else {
        audio_data = utils::JsonUtils::base64_decode(arguments["audio_base64"].get<std::string>());
        if (audio_data.empty()) {
            throw std::runtime_error(
                "audio_base64 decoded to zero bytes. If you meant to pass a "
                "file path, use the `audio_path` argument instead.");
        }
        if (filename.empty()) {
            filename = "audio.wav";
        }
    }

    ensure_loaded_(model);

    json router_request = {
        {"model", model},
        {"file_data", std::move(audio_data)},
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

    std::string transcript = response.value("text", std::string());
    json content = json::array();
    content.push_back(text_content_block(transcript));
    content.push_back(text_content_block(response.dump()));

    return json{
        {"content", std::move(content)},
        {"isError", false},
    };
}

json McpServer::tool_generate_image(const json& arguments) {
    const std::string model = extract_string_arg(arguments, "model");
    const std::string prompt = extract_string_arg(arguments, "prompt");

    // Disk-output mode: MCP image content blocks cost tens of thousands of
    // tokens per image and some clients surface them as opaque resource URIs
    // the agent has to round-trip back to disk. Letting the caller name the
    // destination avoids both. Symmetric with `audio_path`.
    const bool has_output_path = arguments.contains("output_path") &&
                                 arguments["output_path"].is_string();
    const bool has_output_dir  = arguments.contains("output_dir") &&
                                 arguments["output_dir"].is_string();
    if (has_output_path && has_output_dir) {
        throw std::runtime_error(
            "Provide at most one of `output_path` or `output_dir`, not both.");
    }
    std::filesystem::path output_path;
    std::filesystem::path output_dir;
    if (has_output_path) {
        output_path = arguments["output_path"].get<std::string>();
        if (!output_path.is_absolute()) {
            throw std::runtime_error(
                "`output_path` must be an absolute path: " + output_path.string());
        }
    }
    if (has_output_dir) {
        output_dir = arguments["output_dir"].get<std::string>();
        if (!output_dir.is_absolute()) {
            throw std::runtime_error(
                "`output_dir` must be an absolute path: " + output_dir.string());
        }
    }

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
    router_request["response_format"] = "b64_json";

    json response = router_->image_generations(router_request);
    if (response.contains("error")) {
        throw std::runtime_error(response["error"].value("message", "image generation failed"));
    }

    std::vector<std::string> b64_images;
    if (response.contains("data") && response["data"].is_array()) {
        for (const auto& entry : response["data"]) {
            if (!entry.contains("b64_json") || !entry["b64_json"].is_string()) continue;
            b64_images.push_back(entry["b64_json"].get<std::string>());
        }
    }
    if (b64_images.empty()) {
        throw std::runtime_error("Image generation returned no images");
    }

    if (!has_output_path && !has_output_dir) {
        json content = json::array();
        for (const auto& b64 : b64_images) {
            content.push_back({
                {"type", "image"},
                {"data", b64},
                {"mimeType", "image/png"},
            });
        }
        return json{
            {"content", std::move(content)},
            {"isError", false},
        };
    }

    if (has_output_path && b64_images.size() > 1) {
        throw std::runtime_error(
            "`output_path` only supports a single image; use `output_dir` "
            "when n > 1.");
    }

    std::vector<std::filesystem::path> written;
    written.reserve(b64_images.size());
    std::error_code ec;
    if (has_output_dir) {
        std::filesystem::create_directories(output_dir, ec);
        if (ec) {
            throw std::runtime_error(
                "Failed to create output_dir " + output_dir.string() + ": " + ec.message());
        }
    } else {
        auto parent = output_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                throw std::runtime_error(
                    "Failed to create parent of output_path " + parent.string() + ": " + ec.message());
            }
        }
    }

    for (size_t i = 0; i < b64_images.size(); ++i) {
        std::filesystem::path dest = has_output_path
            ? output_path
            : output_dir / ("image_" + std::to_string(i) + ".png");
        std::string bytes = utils::JsonUtils::base64_decode(b64_images[i]);
        if (bytes.empty()) {
            throw std::runtime_error("Decoded image is empty (index " + std::to_string(i) + ")");
        }
        std::ofstream out(dest, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open for writing: " + dest.string());
        }
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out) {
            throw std::runtime_error("Failed to write: " + dest.string());
        }
        written.push_back(std::filesystem::absolute(dest, ec));
        if (ec) written.back() = dest;
    }

    std::string summary = (written.size() == 1)
        ? ("Wrote image to: " + written[0].string())
        : ("Wrote " + std::to_string(written.size()) + " images:");
    json content = json::array();
    content.push_back(text_content_block(summary));
    if (written.size() > 1) {
        for (const auto& p : written) {
            content.push_back(text_content_block(p.string()));
        }
    }
    json paths = json::array();
    for (const auto& p : written) paths.push_back(p.string());
    content.push_back(text_content_block(json{{"paths", std::move(paths)}}.dump()));

    return json{
        {"content", std::move(content)},
        {"isError", false},
    };
}

json McpServer::tool_list_models(const json& arguments) {
    const bool include_available = arguments.value("include_available", true);
    const bool include_suggested = arguments.value("include_suggested", true);

    json loaded = router_->get_all_loaded_models();
    auto downloaded = model_manager_->get_downloaded_models();

    json available = json::array();
    if (include_available) {
        // Only downloaded models — the full registry would blow up client context.
        for (const auto& [model_id, info] : downloaded) {
            json entry = {
                {"model_name", model_id},
                {"recipe", info.recipe},
                {"downloaded", info.downloaded},
                {"suggested", info.suggested},
                {"labels", info.labels},
            };
            if (info.size > 0.0) entry["size_gb"] = info.size;
            available.push_back(std::move(entry));
        }
    }

    json suggested_to_pull = json::array();
    if (include_suggested) {
        for (const auto& [model_id, info] : model_manager_->get_supported_models()) {
            if (!info.suggested) continue;
            if (downloaded.count(model_id)) continue;
            json entry = {
                {"model_name", model_id},
                {"recipe", info.recipe},
                {"labels", info.labels},
            };
            if (info.size > 0.0) entry["size_gb"] = info.size;
            suggested_to_pull.push_back(std::move(entry));
        }
    }

    // Prefer a loaded LLM; fall back to the most recently loaded model of any type.
    std::string recommended;
    for (const auto& m : loaded) {
        if (m.value("type", std::string()) == "llm") {
            recommended = m.value("model_name", std::string());
            break;
        }
    }
    if (recommended.empty() && !loaded.empty()) {
        recommended = loaded.back().value("model_name", std::string());
    }

    json payload = {
        {"loaded", std::move(loaded)},
        {"available", std::move(available)},
        {"suggested_to_pull", std::move(suggested_to_pull)},
        {"recommended_chat_model", recommended},
    };

    std::string summary;
    if (payload["loaded"].empty()) {
        summary = "No models are currently loaded. ";
    } else {
        summary = "Loaded models: ";
        bool first = true;
        for (const auto& m : payload["loaded"]) {
            if (!first) summary += ", ";
            summary += m.value("model_name", std::string());
            summary += " (" + m.value("type", std::string()) + ")";
            first = false;
        }
        summary += ". ";
    }
    if (!recommended.empty()) {
        summary += "Use `" + recommended + "` for chat unless the user asks otherwise.";
    } else if (!payload["suggested_to_pull"].empty()) {
        summary += "No loaded LLM. Suggested models to pull: ";
        bool first = true;
        for (const auto& m : payload["suggested_to_pull"]) {
            if (!first) summary += ", ";
            summary += m.value("model_name", std::string());
            first = false;
        }
        summary += ".";
    }

    return json{
        {"content", json::array({
            text_content_block(summary),
            text_content_block(payload.dump()),
        })},
        {"isError", false},
    };
}

json McpServer::tools_descriptor() {
    return json::array({
        {
            {"name", "lemonade_list_models"},
            {"description",
             "List models known to the Lemonade server. ALWAYS call this "
             "first if you don't already know the exact model name to use "
             "for chat/transcribe/image — passing a wrong name may trigger a "
             "multi-GB download. Returns a summary text block plus a JSON "
             "block with `{loaded, available, suggested_to_pull, "
             "recommended_chat_model}`."},
            {"inputSchema", {
                {"type", "object"},
                {"properties", {
                    {"include_available", {{"type", "boolean"}}},
                    {"include_suggested", {{"type", "boolean"}}},
                }},
            }},
        },
        {
            {"name", "lemonade_chat"},
            {"description",
             "Chat completion against a locally hosted LLM. Pass a `messages` "
             "array (OpenAI chat format) and the model name. If you don't "
             "know which model name to use, call `lemonade_list_models` "
             "first — passing a model that isn't loaded may trigger a "
             "multi-GB download."},
            {"inputSchema", {
                {"type", "object"},
                {"required", json::array({"model", "messages"})},
                {"properties", {
                    {"model",       {{"type", "string"}}},
                    {"messages",    {{"type", "array"}, {"items", {{"type", "object"}}}}},
                    {"temperature", {{"type", "number"}}},
                    {"top_p",       {{"type", "number"}}},
                    {"max_tokens",  {{"type", "integer"}}},
                    {"stop",        {{"description", "stop sequences (string or array)"}}},
                    {"seed",        {{"type", "integer"}}},
                    {"tools",       {{"type", "array"}, {"items", {{"type", "object"}}}}},
                    {"tool_choice", {{"description", "auto | none | required | {type: function, ...}"}}},
                    {"response_format", {{"type", "object"}}},
                    {"chat_template_kwargs", {{"type", "object"},
                                              {"description", "e.g. {\"enable_thinking\": true} to enable reasoning blocks; disabled by default"}}},
                }},
            }},
        },
        {
            {"name", "lemonade_transcribe_audio"},
            {"description",
             "Transcribe an audio clip with a Whisper-class model. The "
             "Lemonade MCP server always runs on the same machine as the "
             "caller, so prefer `audio_path` (an absolute path to a local "
             "audio file: wav, mp3, m4a, ogg, flac, webm). Use "
             "`audio_base64` only when you genuinely have audio bytes in "
             "memory. Exactly one of the two must be provided."},
            {"inputSchema", {
                {"type", "object"},
                {"required", json::array({"model"})},
                {"properties", {
                    {"model",         {{"type", "string"}}},
                    {"audio_path",    {{"type", "string"},
                                       {"description", "Absolute path to a local audio file. Preferred over audio_base64."}}},
                    {"audio_base64",  {{"type", "string"}}},
                    {"filename",      {{"type", "string"}}},
                    {"language",      {{"type", "string"}}},
                    {"prompt",        {{"type", "string"}}},
                    {"response_format", {{"type", "string"},
                                         {"enum", json::array({"json", "text", "srt", "verbose_json", "vtt"})}}},
                    {"temperature",   {{"type", "number"}}},
                }},
            }},
        },
        {
            {"name", "lemonade_generate_image"},
            {"description",
             "Generate one or more images from a text prompt. The Lemonade "
             "MCP server always runs on the same machine as the caller, so "
             "PREFER writing the result directly to disk by passing "
             "`output_path` (single image) or `output_dir` (one or more). "
             "When you do, the tool returns absolute file path(s) as text — "
             "no base64 round-trip and dramatically fewer tokens. Only omit "
             "both arguments when you genuinely need the image inline."},
            {"inputSchema", {
                {"type", "object"},
                {"required", json::array({"model", "prompt"})},
                {"properties", {
                    {"model",  {{"type", "string"}}},
                    {"prompt", {{"type", "string"}}},
                    {"output_path", {{"type", "string"},
                                     {"description", "Absolute path of the PNG file to write. Only valid when n == 1."}}},
                    {"output_dir",  {{"type", "string"},
                                     {"description", "Absolute path of a directory to write image_0.png, image_1.png, ... into."}}},
                    {"size",   {{"type", "string"}}},
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
