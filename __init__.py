import configs.index as conf
import main.server as root
import platform
 
if platform.system() == "Windows":
    import asyncio
 
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

if __name__ == "__main__":
    app = root.application
    app.listen(conf.SERVER_PORT)
    print("Algorithm Server Started in port " + str(conf.SERVER_PORT))
    root.start()
