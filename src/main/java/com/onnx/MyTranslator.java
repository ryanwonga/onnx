package com.onnx;

import java.util.Arrays;
import java.util.List;

import java.util.Arrays;
import java.util.List;

import ai.djl.modality.Classifications;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class MyTranslator implements Translator<IrisFlower, Classifications> {

  private final List<String> synset;

  public MyTranslator() {
    // species name
    synset = Arrays.asList("setosa", "versicolor", "virginica");
  }

  @Override
  public NDList processInput(TranslatorContext ctx, IrisFlower input) {
    float[] data = { input.sepalLength, input.sepalWidth, input.petalLength, input.petalWidth };
    NDArray array = ctx.getNDManager().create(data, new Shape(1, 4));
    return new NDList(array);
  }

  @Override
  public Classifications processOutput(TranslatorContext ctx, NDList list) {
    return new Classifications(synset, list.get(1));
  }

  @Override
  public Batchifier getBatchifier() {
    return null;
  }
}