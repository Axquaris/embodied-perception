import rerun as rr
from ember.zed import ZedLive

def init_rerun():
    rr.init("vis_camera", spawn=True)
    rr.connect_tcp()  # Connect to a remote viewer

def main():
    init_rerun()

    with ZedLive() as zed:
        for frame in zed:
            frame.log_rerun()

    rr.disconnect()
    return 0


if __name__ == "__main__":
    main()
