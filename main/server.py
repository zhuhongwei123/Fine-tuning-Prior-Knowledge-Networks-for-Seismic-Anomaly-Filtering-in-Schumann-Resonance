from abc import ABC
import tornado.ioloop as t_loop
import tornado.web as t_web

import main.schumman as schumman


class MainHandler(t_web.RequestHandler, ABC):
    def get(self):
        self.set_status(404)

    def post(self):
        self.set_status(404)


def start():
    t_loop.IOLoop.current().start()


handlers = [
    (r'/', MainHandler),
    (r'/schumman_analyse', schumman.SchummanAnalyserHandler),
    (r'/mission_state', schumman.MissionFetcherHandler),
]
application = t_web.Application(handlers)
