package com.alphaya.nn.Regularization

import java.io.Serializable

/** Function that computes a penalty cost for a given weight in the network. */
trait RegularizationFunction  extends Serializable {
  def output(weight: Double): Double;

  def der(weight: Double): Double;
}

object L1 extends RegularizationFunction {
  def output(w: Double) = Math.abs(w)

  def der(w: Double) = if (w < 0) -1 else {
    if (w > 0) 1 else 0
  }
}

object L2 extends RegularizationFunction {
  def output(w: Double) = 0.5 * w * w

  def der(w: Double) = w
};

