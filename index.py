import tornado.web
import tornado.ioloop
import os
class uploadImgHandler(tornado.web.RequestHandler):
    def post(self):
        files = self.request.files["imgFile"]
        for f in files:
            fh = open(f"C:/Users/Varadharajan R/Desktop/FYP/img/{f.filename}", "wb")
            fh.write(f.body)
            fh.close()
            output=os.popen("py imgprediction.py","r").read()
            os.remove("C:/Users/Varadharajan R/Desktop/FYP/img/"+f.filename)
        self.write(f"Predicted label of the image {f.filename} is \n "+output)
    def get(self):
        self.render("index.html")

if (__name__ == "__main__"):
    app = tornado.web.Application([
        ("/", uploadImgHandler),
        ("C:/Users/Varadharajan R/Desktop/FYP/img/(.*)", tornado.web.StaticFileHandler, {'path': 'upload'})
    ])

    app.listen(8080)
    print("Listening on port 8080")
    tornado.ioloop.IOLoop.instance().start()