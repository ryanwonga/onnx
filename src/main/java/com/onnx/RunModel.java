package com.onnx;

import java.io.IOException;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;

public class RunModel {
  public static void main(String[] args)
      throws TranslateException, MalformedModelException, ModelNotFoundException, IOException {
    System.setProperty("DJL_CACHE_DIR", "/usr/local/appian/ae/tomcat/apache-tomcat/temp");
    System.setProperty("ENGINE_CACHE_DIR", "/usr/local/appian/ae/tomcat/apache-tomcat/temp");
    String modelUrl = "https://mlrepo.djl.ai/model/tabular/softmax_regression/ai/djl/onnxruntime/iris_flowers/0.0.1/iris_flowers.zip";
    Criteria<IrisFlower,Classifications> criteria = Criteria.builder()
        .setTypes(IrisFlower.class, Classifications.class)
        .optModelUrls(modelUrl)
        .optTranslator(new MyTranslator())
        .optEngine("OnnxRuntime") // use OnnxRuntime engine by default
        .optDevice(Device.cpu())
        .build();
    ZooModel<IrisFlower, Classifications> model = criteria.loadModel();
    Predictor<IrisFlower, Classifications> predictor = model.newPredictor();
    IrisFlower info = new IrisFlower(0.1f, 0.2f, 0.1f, 0.2f);
    System.out.println(predictor.predict(info).toString());
  }
}
