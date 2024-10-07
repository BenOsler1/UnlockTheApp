package com.codedotorg;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;
import javafx.application.Application;
import javafx.stage.Stage;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.IOException;

public class App extends Application {

    public static void main(String[] args) {
        launch();
    }

    @Override
    public void start(Stage primaryStage) {
        Unlock game = new Unlock(primaryStage, 300, 250);
        game.startApp();
        
        // Example usage of TensorFlow model
        try {
            byte[] graphDef = Files.readAllBytes(Paths.get("src/main/resources/model.pb"));
            try (Graph graph = new Graph()) {
                graph.importGraphDef(graphDef);
                try (Session session = new Session(graph)) {
                    float[][] input = {{1.0f, 2.0f, 3.0f}};
                    try (Tensor<Float> inputTensor = Tensors.create(input)) {
                        Tensor<?> outputTensor = session.runner()
                                                        .feed("input_node", inputTensor)
                                                        .fetch("output_node")
                                                        .run()
                                                        .get(0);
                        float[][] output = new float[1][1];
                        outputTensor.copyTo(output);
                        System.out.println("Model output: " + output[0][0]);
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}