import rerun as rr
from ember.zed import ZedRecording

def init_rerun():
    """Initialize Rerun visualization."""
    rr.init("vis_recording", spawn=True)
    rr.connect_tcp()  # Connect to a remote viewer

def main():
    init_rerun()

    svo_path = "data/zed_recordings/HD720_SN33087127_15-44-16.svo2"
    
    with ZedRecording(svo_path) as zed:
        for frame in zed:
            frame.log_rerun()

    rr.disconnect()
    return 0

if __name__ == "__main__":
    main()
