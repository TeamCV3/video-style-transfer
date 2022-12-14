# Application of Artistic Neural Style Transfer Approach for Videos

Artistic neural style transfer (NST) is a famous technique
that reproduces the input image according to the given artistic
style. 
In our project, we expand this technique and try to create a video with transferred painting style.
A video is composed from the frames processed by the NST model. In addition, we apply optical flow in order to reduce noisy movement of the details between the frames.

## Original Video
![Original Video](demos/orig_clip.gif)

## Video with per-frame Transferred Style
![Video with Transferred Style](demos/std_clip.gif)

## Video with Transferred Style and Optical Flow
![Video with Transferred Style](demos/OF_clip.gif)
