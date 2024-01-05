import cv2
from pytube import YouTube

# Replace 'your_youtube_link' with the actual YouTube video link
youtube_link = 'https://www.youtube.com/watch?v=O3DPVlynUM0'

# Function to initialize video capture from YouTube link
def initialize_youtube_video(youtube_link):
    # Download the YouTube video
    youtube = YouTube(youtube_link)
    video_stream = youtube.streams.filter(file_extension="mp4").first()
    video_stream.download("youtube_video.mp4")

    # Initialize video capture
    cap = cv2.VideoCapture("youtube_video.mp4")
    return cap

# Main function to show frames from the live YouTube video
def show_youtube_frames(youtube_link):
    # Initialize video capture
    cap = initialize_youtube_video(youtube_link)

    while True:
        # Read a frame
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

        # Display the frame
        cv2.imshow("YouTube Video", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()

# Run the function
show_youtube_frames(youtube_link)
