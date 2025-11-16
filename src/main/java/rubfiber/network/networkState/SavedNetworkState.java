package rubfiber.network.networkState;

import com.google.gson.annotations.Expose;

import java.util.List;

public class SavedNetworkState {
    @Expose
    public int hiddenLayersWidth;
    @Expose
    public int hiddenLayersDepth;
    @Expose
    public int inputLayer;
    @Expose
    public int outputLayer;

    @Expose
    public List<List<Double>> biases;
    @Expose
    public List<List<List<Double>>> weights;
}