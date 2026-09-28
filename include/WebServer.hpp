#pragma once
#include <string>
#include <functional>

class WebServer {
public:
    WebServer(const std::string& config_path, const std::string& web_root_path);
    ~WebServer();

    // 启动非阻塞 Web 服务线程
    void Start(int port);
    
    // 停止服务
    void Stop();

    // 注册配置变更回调函数
    void SetOnConfigUpdated(std::function<void()> callback) {
        on_config_updated_ = callback;
    }

    // 注册重启引擎回调函数
    void SetOnRestartEngine(std::function<void()> callback) {
        on_restart_engine_ = callback;
    }

    // 注册一个 Provider，用于把当前模型支持的英文/中文类名透传给前端。
    // Provider 返回的 vector 是引用，调用方需保证生命周期长于 WebServer。
    using ClassNamesProvider = std::function<const std::vector<std::string>&()>;
    void SetClassNamesProvider(ClassNamesProvider en_provider, ClassNamesProvider cn_provider) {
        class_names_en_provider_ = std::move(en_provider);
        class_names_cn_provider_ = std::move(cn_provider);
    }

private:
    void InitRoutes();

    std::string config_path_;
    std::string web_root_path_;
    void* server_ptr_; // 类型擦除，隐藏 httplib 依赖
    std::function<void()> on_config_updated_ = nullptr;
    std::function<void()> on_restart_engine_ = nullptr;
    ClassNamesProvider class_names_en_provider_;
    ClassNamesProvider class_names_cn_provider_;
};
