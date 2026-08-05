upstream app {
	server app:${WEB_APP_PORT};
}

server {
	server_name localhost;
	listen ${WEB_SERVER_PORT};
	proxy_request_buffering off;
	client_max_body_size 4096m;
	location /oidc {
		proxy_pass http://app/oidc;
		# Set `proxy_set_header Host` so that the OIDC callback will look like 
		# http://localhost:${API_PROXY_PORT} in the case of local development
		proxy_set_header Host ${DOLLAR}host:${WEB_SERVER_PORT} ;
	}

	location /(.+)$  {
		proxy_pass http://app/${DOLLAR}1;
	}

	location / {
		proxy_pass http://app/;
		rewrite ^/$ /static/index.html last;
	}

	location /static/ {
		alias /static/;
	}

	location /robots.txt {
		alias /static/robots.txt;
	}

	location /api {
		proxy_pass http://app/api;
		proxy_set_header Host ${DOLLAR}host:${WEB_SERVER_PORT} ;

		# See the reasoning above for customizing the header
		# Correct download url showing up in api serialized output
	}
}
