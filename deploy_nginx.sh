#!/bin/bash
# 阿里云服务器部署脚本 - Nginx方案

# 1. 更新系统并安装Nginx
echo "=== 安装Nginx ==="
sudo apt update
sudo apt install -y nginx

# 2. 创建网站目录
echo "=== 创建网站目录 ==="
sudo mkdir -p /var/www/langchain-tutorial
sudo chown -R $USER:$USER /var/www/langchain-tutorial

# 3. 配置Nginx
echo "=== 配置Nginx ==="
sudo tee /etc/nginx/sites-available/langchain-tutorial > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    root /var/www/langchain-tutorial;
    index index.html;

    # 启用gzip压缩
    gzip on;
    gzip_types text/html text/css application/javascript;

    # 缓存静态文件
    location ~* \.(html|css|js|png|jpg|jpeg|gif|ico)$ {
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # 安全头部
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
}
EOF

# 4. 启用站点配置
sudo ln -sf /etc/nginx/sites-available/langchain-tutorial /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 5. 配置防火墙
echo "=== 配置防火墙 ==="
sudo ufw allow 'Nginx HTTP'
sudo ufw allow OpenSSH
sudo ufw --force enable

echo "=== Nginx部署完成! ==="
echo "请将HTML文件上传到: /var/www/langchain-tutorial"
echo "访问地址: http://你的服务器IP"
