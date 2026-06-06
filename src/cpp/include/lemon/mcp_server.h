#pragma once

#include <functional>
#include <memory>
#include <string>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "model_manager.h"
#include "router.h"

namespace lemon {

using json = nlohmann::json;

// Model Context Protocol (MCP) gateway server.
//
// Exposes Lemonade's inference capabilities (chat, embeddings, audio
// transcription, text-to-speech, image generation) as MCP tools served over a
// single HTTP endpoint: POST /mcp. The wire protocol is JSON-RPC 2.0 over
// HTTP (the "Streamable HTTP" transport from MCP spec 2025-06-18).
//
// IMPORTANT: /mcp is an INTENTIONAL EXCEPTION to the quad-prefix invariant
// (AGENTS.md Critical Invariant #1). The MCP spec mandates a single endpoint
// URL per server, so registering it under /api/v0/mcp, /api/v1/mcp, /v0/mcp,
// /v1/mcp would not match what MCP clients expect. Precedent: Ollama uses
// /api/* (its own protocol prefix) and Anthropic uses POST /api/messages.
class McpServer : public std::enable_shared_from_this<McpServer> {
public:
    // Callback used to lazily load (and, if needed, download) a model before
    // a tool dispatches to the router. Matches the signature used by
    // CollectionOrchestrator so that Server::auto_load_model_if_needed can be
    // passed in directly.
    using EnsureLoadedFn = std::function<void(const std::string&)>;

    McpServer(Router* router, ModelManager* model_manager, EnsureLoadedFn ensure_loaded);

    // Registers POST /mcp and GET /mcp (405 Method Not Allowed) on the given
    // httplib server. Must be called on a shared_ptr instance — the handler
    // lambdas capture a shared_from_this() to keep this object alive for the
    // lifetime of the server.
    void register_routes(httplib::Server& server);

    // Parse a JSON-RPC 2.0 request body (single object or batch array) and
    // produce the response body. Returns an empty string if the request
    // contained only notifications (per JSON-RPC, no response is sent).
    std::string handle_request_body(const std::string& body);

private:
    // Dispatch a single JSON-RPC message. Returns json() (a null value) if
    // the message is a notification (has no `id` field) — the caller filters
    // those out of the response.
    json dispatch_message(const json& message);

    // Method handlers
    json handle_initialize(const json& params, const json& id);
    json handle_tools_list(const json& id);
    json handle_tools_call(const json& params, const json& id);
    json handle_ping(const json& id);

    // Tool dispatchers — each takes the `arguments` object from a tools/call
    // request and returns a JSON-RPC `result` payload (a `{content, isError}`
    // object). On invalid arguments, throws std::runtime_error; the caller
    // wraps the message in an isError=true result block.
    json tool_chat(const json& arguments);
    json tool_embed(const json& arguments);
    json tool_transcribe_audio(const json& arguments);
    json tool_generate_speech(const json& arguments);
    json tool_generate_image(const json& arguments);
    json tool_list_models(const json& arguments);

    // Static descriptors used by `tools/list`. Each tool's input JSON Schema
    // is embedded as a literal here to keep the dependency graph minimal.
    static json tools_descriptor();

    // JSON-RPC envelope builders
    static json make_error_response(const json& id, int code, const std::string& message);
    static json make_success_response(const json& id, json result);

    // MCP content block helpers
    static json text_content_block(const std::string& text);

    Router* router_;
    ModelManager* model_manager_;
    EnsureLoadedFn ensure_loaded_;
};

}  // namespace lemon
