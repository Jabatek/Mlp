package com.alphaya.nn.Activation

import java.io.Serializable

trait ActivationFunction  extends Serializable{
  def output(input: Double): Double

  def der(input: Double): Double
}

object RELU extends ActivationFunction {
  def output(x: Double): Double = Math.max(0, x)

  def der(x: Double): Double = if (x <= 0) 0 else 1

}

object TANH extends ActivationFunction {
  def output(x: Double): Double = {
    Math.tanh(x)
  }

  def der(x: Double): Double = {
    var output =Math.tanh(x)
    1-output *output

  }
}

object SIGMOID extends ActivationFunction {
  def output(x: Double) = {
    1 / (1 + Math.exp(-x))
  }: Double

  def der(x: Double) = {
    var output = SIGMOID.output(x);
    output * (1 - output)
  }: Double
}



object LINEAR extends ActivationFunction {
  def output(x: Double): Double = x

  def der(x: Double): Double = 1
}

object TRIPLET extends ActivationFunction {
  def output(x: Double): Double = if (x > 0.5) 1 else if (x < -0.5) -1 else 0

  def der(x: Double): Double = 1
}

object BINARY extends ActivationFunction {
  def output(x: Double): Double = if (x > 0) 1 else if (x < 0) -1 else 0

  def der(x: Double): Double = 1
}