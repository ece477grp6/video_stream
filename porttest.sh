(echo >/dev/tcp/{host}/{port}) &>/dev/null && echo "open" || echo "close"
(echo >/dev/udp/{host}/{port}) &>/dev/null && echo "open" || echo "close"
