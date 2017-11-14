package com.alphaya.nn.Error

import java.io.Serializable

trait ErrorFunction  extends Serializable{
  def error(output: Double, target: Double): Double;

  def der(output: Double, target: Double): Double;
}

object SQUARE extends ErrorFunction {
  def error(output: Double, target: Double) = 0.5 * Math.pow(output - target, 2)

  def der(output: Double, target: Double) = output - target

}
