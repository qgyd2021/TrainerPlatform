# -*- encoding=UTF-8 -*-
from server.flask_server.route_wrap.common_route_wrap import common_route_wrap


@common_route_wrap
def heart_beat():
    return 'OK'
