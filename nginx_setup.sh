#!/bin/bash
cat > /etc/nginx/sites-available/sentiment-api << 'NGINXEOF'
server {
    listen 80;
    server_name sentimentanalysis.sameertripathi.dev;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }
}
NGINXEOF

ln -sf /etc/nginx/sites-available/sentiment-api /etc/nginx/sites-enabled/sentiment-api
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx && echo "NGINX_OK"

# Get SSL certificate
certbot --nginx -d sentimentanalysis.sameertripathi.dev --non-interactive --agree-tos -m tripathisam2204@gmail.com && echo "SSL_OK"
