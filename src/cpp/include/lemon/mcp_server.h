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

// MCP gateway: JSON-RPC 2.0 over HTTP (Streamable HTTP transport, spec 2025-06-18).
//
// /mcp is an INTENTIONAL EXCEPTION to the quad-prefix invariant:
// the MCP spec mandates a single endpoint URL.
class McpServer : public std::enable_shared_from_this<McpServer> {
public:
    using EnsureLoadedFn = std::function<void(const std::string&)>;

    McpServer(Router* router, ModelManager* model_manager, EnsureLoadedFn ensure_loaded);

    // Must be called on a shared_ptr instance — handlers capture shared_from_this().
    void register_routes(httplib::Server& server);

    std::string handle_request_body(const std::string& body);

private:
    json dispatch_message(const json& message);

    json handle_initialize(const json& params, const json& id);
    json handle_tools_list(const json& id);
    json handle_tools_call(const json& params, const json& id);
    json handle_ping(const json& id);

    json tool_chat(const json& arguments);
    json tool_transcribe_audio(const json& arguments);
    json tool_generate_image(const json& arguments);
    json tool_omni(const json& arguments);
    json tool_list_models(const json& arguments);

    static json tools_descriptor();

    static json make_error_response(const json& id, int code, const std::string& message);
    static json make_success_response(const json& id, json result);
    static json text_content_block(const std::string& text);

    Router* router_;
    ModelManager* model_manager_;
    EnsureLoadedFn ensure_loaded_;
};

}  // namespace lemon
