#!/bin/bash

# 自动化启动项目脚本
# 先启动 sample_audio.py，然后启动 lightrag_bmodel2.py
# 所有输出保存到 ./output.txt
#firejail --noprofile --net=none --protocol=unix,inet ./start_project.sh

set -e  # 遇到错误立即退出

# 创建输出文件并重定向所有输出
exec > >(tee -a ./output.txt)
exec 2>&1

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 脚本开始时添加时间戳
echo "============================================"
echo "脚本启动时间: $(date)"
echo "输出文件: $(pwd)/output.txt"
echo "============================================"

# 检查依赖
check_dependencies() {
    log_step "检查依赖环境..."
    
    # 检查目录
    if [ ! -d "/data/whisper-TPU_py" ]; then
        log_error "whisper-TPU_py 目录不存在"
        exit 1
    fi
    
    if [ ! -d "/data/LightRAG/examples" ]; then
        log_error "LightRAG/examples 目录不存在"
        exit 1
    fi
    
    # 检查虚拟环境
    if [ ! -d "/data/whisper-TPU_py/.venv" ]; then
        log_error "whisper-TPU_py/.venv 虚拟环境不存在"
        exit 1
    fi
    
    # 检查虚拟环境
    if [ ! -d "/data/LightRAG/.venv310" ]; then
        log_error "LightRAG/.venv310 虚拟环境不存在"
        exit 1
    fi

    log_info "依赖检查通过"
}

# 检查端口是否被占用
check_port() {
    local port=$1
    if netstat -tuln | grep -q ":$port "; then
        log_warn "端口 $port 已被占用"
        return 1
    fi
    return 0
}

# 等待API服务启动
wait_for_api() {
    log_step "等待 API 服务启动..."
    local attempt=0
    
    while true; do  # 改为无限循环
        if curl -s "http://localhost:8899/status" > /dev/null 2>&1; then
            log_info "API 服务已启动并可用"
            return 0
        fi
        
        if [ $((attempt % 30)) -eq 0 ]; then  # 每30秒显示一次等待信息
            log_info "已等待 $attempt 秒，继续等待API服务..."
        fi
        
        echo -n "."
        sleep 1
        ((attempt++))
    done
}

# 启动 lightrag_bmodel2.py (后台运行)
start_lightrag() {
    log_step "启动 lightrag_bmodel2.py (后台运行)..."
    
    # 激活 lightrag 环境并启动，输出也重定向到主输出文件
    (
        cd /data/LightRAG
        source .venv310/bin/activate
        cd /data/LightRAG/examples/
        echo "========== LightRAG 启动输出 =========="
        python lightrag_bmodel2.py
        echo "========== LightRAG 结束输出 =========="
    ) &
    
    LIGHTRAG_PID=$!
    echo $LIGHTRAG_PID > /tmp/lightrag.pid
    
    log_info "lightrag_bmodel2.py 启动成功 (PID: $LIGHTRAG_PID)"
    log_info "LightRAG 输出已合并到主输出文件"
}

# 清理函数
cleanup() {
    echo ""
    log_step "正在清理进程..."
    echo "清理时间: $(date)"
    
    if [ -f "/tmp/lightrag.pid" ]; then
        LIGHTRAG_PID=$(cat /tmp/lightrag.pid)
        if kill -0 $LIGHTRAG_PID 2>/dev/null; then
            log_info "停止 lightrag_bmodel2.py (PID: $LIGHTRAG_PID)"
            kill $LIGHTRAG_PID
        fi
        rm -f /tmp/lightrag.pid
    fi
    
    log_info "清理完成"
    echo "============================================"
    echo "脚本结束时间: $(date)"
    echo "============================================"
}

# 主函数
main() {
    case "${1:-start}" in
        "start")
            log_info "开始启动项目..."
            echo "============================================"
            
            # 检查依赖
            check_dependencies
            
            # 检查端口8899是否可用
            if ! check_port 8899; then
                echo -n "端口8899被占用，是否杀死占用进程？(y/n): "
                read kill_process < /dev/tty  # 从终端读取输入
                if [ "$kill_process" = "y" ] || [ "$kill_process" = "Y" ]; then
                    log_info "杀死占用端口8899的进程..."
                    sudo fuser -k 8899/tcp 2>/dev/null || true
                    sleep 2
                else
                    log_error "端口被占用，无法启动服务"
                    exit 1
                fi
            fi
            
            # 启动 sample_audio.py (前台运行)
            log_step "启动 sample_audio.py (前台显示输出)..."
            cd /data/whisper-TPU_py
            source .venv/bin/activate
            cd bmwhisper/
            
            log_info "正在启动 sample_audio.py..."
            echo "=========================================="
            echo "      Sample Audio 前台运行中..."
            echo "=========================================="
            
            # 设置清理函数在退出时执行
            trap cleanup SIGINT SIGTERM EXIT
            
            # 在后台监控API状态并启动LightRAG
            (
                # 等待API服务启动
                if wait_for_api; then
                    log_info "API服务已就绪，启动 LightRAG..."
                    start_lightrag
                    
                    log_info "所有服务启动完成！"
                    log_info "sample_audio.py: 前台运行 (显示所有输出)"
                    log_info "lightrag_bmodel2.py: 后台运行"
                    log_info "所有输出保存到: $(pwd)/output.txt"
                    log_warn "按 Ctrl+C 停止所有服务"
                else
                    log_error "API 服务启动失败，无法启动 LightRAG"
                fi
            ) &
            
            # sample_audio.py 在前台运行，显示所有输出
            echo "========== Sample Audio 启动输出 =========="
            python3 sample_audio.py
            echo "========== Sample Audio 结束输出 =========="
            ;;
            
        "stop")
            log_info "停止所有服务..."
            cleanup
            
            # 尝试停止可能运行的 sample_audio.py
            if pgrep -f "sample_audio.py" > /dev/null; then
                log_info "停止 sample_audio.py..."
                pkill -f "sample_audio.py"
            fi
            ;;
            
        "status")
            log_info "检查服务状态..."
            echo "状态检查时间: $(date)"
            
            # 检查 sample_audio.py
            if curl -s "http://localhost:8899/status" > /dev/null 2>&1; then
                log_info "sample_audio.py: 运行中"
                curl -s "http://localhost:8899/status"
            else
                log_warn "sample_audio.py: 未运行"
            fi
            
            # 检查 lightrag 进程
            if [ -f "/tmp/lightrag.pid" ]; then
                LIGHTRAG_PID=$(cat /tmp/lightrag.pid)
                if kill -0 $LIGHTRAG_PID 2>/dev/null; then
                    log_info "lightrag_bmodel2.py PID: $LIGHTRAG_PID (运行中)"
                else
                    log_warn "lightrag_bmodel2.py PID: $LIGHTRAG_PID (已停止)"
                fi
            else
                log_warn "lightrag_bmodel2.py: 未启动"
            fi
            
            # 检查输出文件
            if [ -f "./output.txt" ]; then
                log_info "输出文件: $(pwd)/output.txt ($(wc -l < ./output.txt) 行)"
            else
                log_warn "输出文件不存在"
            fi
            ;;
            
        "logs")
            log_info "查看输出文件..."
            if [ -f "./output.txt" ]; then
                echo "=========================================="
                echo "      实时输出 (按 Ctrl+C 停止查看)"
                echo "=========================================="
                tail -f ./output.txt
            else
                log_warn "输出文件不存在，请先启动服务"
            fi
            ;;
            
        "help"|"-h"|"--help")
            echo "用法: $0 [命令]"
            echo ""
            echo "命令:"
            echo "  start    启动所有服务 (默认)"
            echo "  stop     停止所有服务"
            echo "  status   检查服务状态"
            echo "  logs     查看实时输出"
            echo "  help     显示帮助信息"
            echo ""
            echo "说明:"
            echo "  - sample_audio.py 前台运行"
            echo "  - lightrag_bmodel2.py 后台运行"
            echo "  - 所有输出保存到 ./output.txt"
            echo "  - 等待API服务就绪后自动启动LightRAG"
            echo "  - 按 Ctrl+C 停止所有服务"
            ;;
            
        *)
            log_error "未知命令: $1"
            echo "使用 '$0 help' 查看可用命令"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"