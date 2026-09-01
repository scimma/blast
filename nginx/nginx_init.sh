set -e

export DOLLAR="$"
envsubst < /etc/nginx/conf.d/default.conf.tpl > /etc/nginx/conf.d/default.conf
cat /etc/nginx/conf.d/default.conf
