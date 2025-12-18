#!/bin/bash
# Docker 权限修复脚本

echo "==========================================="
echo "Docker 权限修复脚本"
echo "==========================================="
echo ""

# 检查是否以 root 运行
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  警告: 请不要以 root 用户运行此脚本"
    echo "请以普通用户运行，脚本会提示输入 sudo 密码"
    exit 1
fi

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker 未安装"
    echo "请先安装 Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# 检查 Docker 服务是否运行
if ! systemctl is-active --quiet docker 2>/dev/null; then
    echo "启动 Docker 服务..."
    sudo systemctl start docker
    sudo systemctl enable docker
    echo "✅ Docker 服务已启动"
    echo ""
fi

# 检查用户是否已在 docker 组
if groups | grep -q docker; then
    echo "✅ 用户 $USER 已在 docker 组中"
    echo ""
    echo "如果仍然遇到权限问题，请尝试："
    echo "1. 重新登录"
    echo "2. 或执行: newgrp docker"
    echo ""
else
    echo "将用户 $USER 添加到 docker 组..."
    sudo usermod -aG docker $USER
    echo "✅ 用户已添加到 docker 组"
    echo ""
    echo "⚠️  重要: 需要重新登录或执行以下命令使更改生效："
    echo "  newgrp docker"
    echo ""
    echo "或者直接执行："
    echo "  newgrp docker"
    echo ""
    read -p "是否现在执行 newgrp docker？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "执行 newgrp docker..."
        newgrp docker
        echo "✅ 权限已更新"
    else
        echo "请手动执行: newgrp docker"
    fi
fi

echo ""
echo "==========================================="
echo "验证 Docker 权限..."
echo "==========================================="

# 测试 Docker 权限
if docker ps &> /dev/null; then
    echo "✅ Docker 权限正常"
    echo ""
    echo "可以运行: ./docker/build_and_run.sh"
else
    echo "❌ Docker 权限仍有问题"
    echo ""
    echo "请尝试："
    echo "1. 重新登录"
    echo "2. 执行: newgrp docker"
    echo "3. 或使用: sudo ./docker/build_and_run.sh"
fi

