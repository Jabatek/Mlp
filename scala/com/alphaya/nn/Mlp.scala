
package com.alphaya.nn

import com.alphaya.nn.Activation._
import com.alphaya.nn.Error._
import com.alphaya.nn.Regularization._
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import java.io.Serializable


  class Link(var source: Node, var dest: Node,val optimizer: Optimizer,  initZero: Boolean)  extends Serializable{

  // Initialize the weights.
  var weight: Double = Random.nextDouble() - 0.5
  /** Accumulated error derivative since the last update. */
  var accErrorDer: Double = 0
  /** Number of accumulated derivatives since the last update. */
  var numAccumulatedDers = 0

  if (initZero) {
    weight = 0;
  }
    // regularization switch
    var isDead = false
}

class Node( val activation: ActivationFunction)  extends Serializable{

  /** List of input links. */
  var inputLinks: ArrayBuffer[Link] = ArrayBuffer[Link]()
  /** List of output links. */
  var outputs: ArrayBuffer[Link] = ArrayBuffer[Link]()

  // w*x
  var totalInput: Double = 0

  // activation(w*x)
  var output: Double = 0
  /** Error derivative with respect to this node's output. */
  var outputDer: Double = 0;

  /** Error derivative with respect to this node's total input. */
  var inputDer: Double = 0;


  /** Recomputes the node's output and returns it. */
  def updateOutput(): Double = {
    // Stores total input into the node.
    totalInput = inputLinks(0).weight  //bias

    for (il <-1 until inputLinks.length) {

      totalInput += inputLinks(il).weight * inputLinks(il).source.output

    }

    output = activation.output(totalInput)
    output
  }

}



class Mlp (networkShape: Array[Int], activation: ActivationFunction, outputActivation: ActivationFunction,
            gOptimizer: GOptimizer,  regularization: RegularizationFunction, initZero: Boolean) extends Serializable {

  var iter:Long=1

    /** List of layers, with each layer being a list of nodes. */
    var network= ArrayBuffer[ArrayBuffer[Node]]()

    for (layerIdx <- 0 until networkShape.length) {
      var isOutputLayer = layerIdx == networkShape.length - 1
      var isInputLayer = layerIdx == 0

      var currentLayer: ArrayBuffer[Node] = ArrayBuffer[Node]()
      network += currentLayer

      var numNodes = networkShape(layerIdx)

      for (i <- 0 until numNodes) {
        var node = new Node( if (isOutputLayer) outputActivation else activation)
        currentLayer += node
        if (layerIdx >0) {
          // Add links from nodes in the previous layer to this node.

          node.inputLinks += new Link(null, node,gOptimizer.generate(),  initZero)//bias
          node.inputLinks.last.weight=0.1
          //other links
          for (prevNode <- network(layerIdx - 1)) {
            var link = new Link(prevNode, node,gOptimizer.generate(),  initZero)
            prevNode.outputs += link
            node.inputLinks += link
          }
        }
      }

    }


  def forwardProp( inputs: Array[Double]): Double = {
    var inputLayer = network(0);
    // Update the input layer.
    for (i <- 0 until inputLayer.length) {
      var node = inputLayer(i)
      node.output = inputs(i)
    }
    for (layerIdx <- 1 until network.length) {
      val currentLayer = network(layerIdx)
      // Update all the nodes in this layer.
      for (node <- currentLayer) {
        node.updateOutput()
      }
    }

    network.last(0).output
  }


  /**
    * Runs a backward propagation using the provided target and the
    * computed output of the previous call to forward propagation.
    * This method modifies the internal state of the network - the error
    * derivatives with respect to each node, and each weight
    * in the network.
    */


  def backProp( target: Double, errorFunc: ErrorFunction) :Unit= {
    // The output node is a special case. We use the user-defined error
    // function for the derivative.

      var outputNode = network.last(0)
      outputNode.outputDer = errorFunc.der(outputNode.output, target)
      // Go through the layers backwards.
      for (layerIdx <- (network.length - 1) to 1 by -1) {
        var currentLayer = network(layerIdx)
        for (node <- currentLayer) {
          // Compute the error derivative of each node with respect to:
          // 1) its total input
          // 2) each of its input weights.
          node.inputDer = node.outputDer * activation.der(node.totalInput)
          // Error derivative with respect to each weight coming into the node.


          var link = node.inputLinks(0)//bias
              link.accErrorDer += node.inputDer
              link.numAccumulatedDers += 1

          for (i <-1 until node.inputLinks.length) {
            link=node.inputLinks(i)
            if (!link.isDead) {
              link.accErrorDer += node.inputDer * link.source.output
              link.numAccumulatedDers += 1
            }
          }
        }
        if (layerIdx != 1) {
          var prevLayer = network(layerIdx - 1)
          for (node <- prevLayer) {
            // Compute the error derivative with respect to each node's output.
            node.outputDer = 0;
            for (olink <- node.outputs) {
              node.outputDer += olink.weight * olink.dest.inputDer;
            }

          }
        }
      }
    }

  def backProp( targets: Array[Double], errorFunc: ErrorFunction) :Unit= {

    for (i <- 1 until network.last.length) {
      var outputSNode = network.last(i)
      var target = targets(i)
      outputSNode.outputDer = errorFunc.der(outputSNode.output, target)
    }

    backProp( targets(0), errorFunc)
  }

  def updateWeights( learningRate: Double, regularizationRate: Double) = {
    iter+=1
    for (layerIdx <- 1 until network.length) {
      var currentLayer = network(layerIdx);
      for (node <- currentLayer) {

        // Update the weights coming into this node.
        for (link <- node.inputLinks) {

          if (!link.isDead) {
            var regulDer: Double = if (regularization != null) regularization.der(link.weight) else 0
            if (link.numAccumulatedDers > 0) {
              // Update the weight based on dE/dw.
              link.weight=link.optimizer.optimize(link.weight,link.accErrorDer,learningRate,iter )/link.numAccumulatedDers
              // Further update the weight based on regularization.
              var newLinkWeight = link.weight - regulDer * (learningRate * regularizationRate)
              if (regularization == L1 && link.weight * newLinkWeight < 0) {
                // The weight crossed 0 due to the regularization term. Set it to 0.
                link.weight = 0
                link.isDead = true
              } else {
                link.weight = newLinkWeight
              }
              link.accErrorDer = 0
              link.numAccumulatedDers = 0
            }
          }

        }
      }
    }
  }

}




