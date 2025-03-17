import argparse
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train MLFFNN with specified hyperparameters.")
    parser.add_argument("epochs", type=int, nargs="?", default=10, help="Number of epochs (default: 10)")
    parser.add_argument("num_hidden_layers", type=int, nargs="?", default=5, help="Number of hidden layers (default: 5)")
    parser.add_argument("hidden_size", type=int, nargs="?", default=128, help="Size of hidden layers (default: 128)")
    parser.add_argument("weight_decay", type=float, nargs="?", default=0.0005, help="Weight decay (default: 0.0005)")
    parser.add_argument("learning_rate", type=float, nargs="?", default=0.0001, help="Learning rate (default: 0.0001)")
    parser.add_argument("optimizer", type=str, nargs="?", default="adam", help="Optimizer (default: adam)")
    parser.add_argument("batch_size", type=int, nargs="?", default=16, help="Batch size (default: 16)")
    parser.add_argument("weight_init", type=str, nargs="?", default="xavier", help="Weight initialization method (default: xavier)")
    parser.add_argument("activation", type=str, nargs="?", default="tanh", help="Activation function (default: tanh)")
    
    return parser.parse_args()
