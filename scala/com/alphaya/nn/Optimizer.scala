package com.alphaya.nn

import java.io.Serializable

trait GOptimizer  extends Serializable{
  var rate:Double=.0001
  def generate(): Optimizer =
  {
    null
  }
}

trait Optimizer  extends Serializable{

  def optimize(w:Double,dw:Double,rate:Double,iter:Long): Double =
  {
  return w-rate*dw
  }
}

object sdg extends GOptimizer with Optimizer
{
  override  def generate(): Optimizer ={
    return this
  }
}


class Mom extends   Optimizer{

  val beta1 = .9
  var m=.0

  override def optimize(w:Double,dw:Double,rate:Double,iter:Long): Double =
  {
    m=beta1 * m + GMom.rate* dw
    m
  }
}

object GMom extends GOptimizer {

  override  def generate(): Optimizer ={
    new Adam()
  }
}

class Adam extends   Optimizer{

  val beta1 = .9
  val beta2 = .999
  val eps=.00000008
  var m=.0
  var r=.0

  override def optimize(w:Double,dw:Double,rate: Double,iter:Long): Double =
  {

    m=beta1 * m + (1 - beta1) * dw
    r=beta2 * r + (1 - beta2) * dw*dw

    val m_k = m / (1 - math.pow(beta1,iter))
    val r_k = r / (1 - math.pow(beta2,iter))
//    println(iter,m_k,math.pow(beta1,iter))
    w-rate *m_k / (math.sqrt(r_k) + eps)

  }
}

object GAdam extends GOptimizer {

  override  def generate(): Optimizer ={
     new Adam()
  }
}




