import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')

from gi.repository import Gst, GstRtspServer, GLib

Gst.init(None)

server = GstRtspServer.RTSPServer()

# 🔥 IMPORTANT: allow remote connections
server.set_address("0.0.0.0")
server.set_service("8554")

factory = GstRtspServer.RTSPMediaFactory()
factory.set_launch(
    '( v4l2src device=/dev/video0 ! videoconvert ! '
    'x264enc tune=zerolatency speed-preset=ultrafast bitrate=800 key-int-max=30 ! '
    'rtph264pay name=pay0 pt=96 config-interval=1 )'
)

factory.set_shared(True)

# Launch using ecternal camera

# --- 480p ----
# factory.set_launch(
#     '( v4l2src device=/dev/video2 ! '
#     'video/x-raw,framerate=30/1,width=640,height=480 ! '
#     'videoconvert ! '
#     'x264enc tune=zerolatency speed-preset=ultrafast bitrate=800 key-int-max=30 ! '
#     'rtph264pay name=pay0 pt=96 config-interval=1 )'
# )

# -- 720p -----
factory.set_launch(
    '( v4l2src device=/dev/video2 ! '
    'image/jpeg,width=1280,height=720,framerate=30/1 ! '
    'jpegdec ! videoconvert ! '
    'x264enc tune=zerolatency speed-preset=veryfast bitrate=3000 key-int-max=30 ! '
    'rtph264pay name=pay0 pt=96 config-interval=1 )'
)


# factory.set_launch(
#     '( v4l2src device=/dev/video2 ! '
#     'image/jpeg,width=1920,height=1080,framerate=30/1 ! '
#     'jpegdec ! videoconvert ! '
#     'x264enc tune=zerolatency speed-preset=ultrafast bitrate=6000 key-int-max=30 ! '
#     'rtph264pay name=pay0 pt=96 config-interval=1 )'
# )


mounts = server.get_mount_points()
mounts.add_factory("/stream", factory)

server.attach(None)

print("RTSP camera running at:")
print("  rtsp://127.0.0.1:8554/stream")

GLib.MainLoop().run()
