import logging
import time
import traceback

import jsonschema

from server.exception import ExpectedError
from toolbox.logging.misc import json_2_str

logger = logging.getLogger('server')


result_schema = {
    'type': 'object',
    'required': ['result', 'debug'],
    'properties': {
        'result': {},
        'debug': {}
    },
    'additionalProperties': False

}


def common_route_wrap(f):
    def inner(*args, **kwargs):
        begin = time.time()
        try:
            ret = f(*args, **kwargs)
            try:
                jsonschema.validate(ret, result_schema)
                debug = ret['debug']
                result = ret['result']
            except jsonschema.exceptions.ValidationError as e:
                debug = None
                result = ret

            response = {
                'status_code': 60200,
                'result': result,
                'debug': debug,
                'message': 'success',
                'detail': None
            }
            status_code = 200
        except ExpectedError as e:
            response = {
                'status_code': e.status_code,
                'result': None,
                'message': e.message,
                'detail': e.detail,
                'traceback': e.traceback,
            }
            status_code = 400

        except Exception as e:
            response = {
                'status_code': 60500,
                'result': None,
                'message': str(e),
                'detail': None,
                'traceback': traceback.format_exc(),
            }
            status_code = 500

        cost = time.time() - begin
        response['time_cost'] = round(cost, 4)

        abstract_response = json_2_str(response)
        if 'traceback' in response:
            abstract_response['traceback'] = response['traceback']

        logger.info('response: {}'.format(abstract_response))
        # logger.info('response: {}'.format(json.dumps(response, ensure_ascii=False)))

        return response, status_code
    return inner
