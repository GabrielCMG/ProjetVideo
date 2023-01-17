
vidReader = webcam;

% opticFlow = opticalFlowHS("VelocityDifference", 0.01);
opticFlow = opticalFlowLK('NoiseThreshold',0.01);

h = figure;
movegui(h);
hViewPanel = uipanel(h,'Position',[0 0 1 1],'Title','Plot of Optical Flow Vectors');
hPlot = axes(hViewPanel);

while 1
    tic
    frameRGB = snapshot(vidReader);
    frameGray = im2gray(frameRGB);
    toc
    flow = estimateFlow(opticFlow, frameGray);
    imshow(frameRGB)
    hold on
    plot(flow,'DecimationFactor',[5 5],'ScaleFactor',10,'Parent',hPlot);
    hold off
end