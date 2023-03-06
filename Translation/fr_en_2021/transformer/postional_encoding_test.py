# ensures the positional encoding works as intended
from transformer.positional_encoding import positional_encoding
import matplotlib.pyplot as plt


def test():
    """Performs the test for the positonal encoding"""
    pos_encoding = positional_encoding(50, 512)
    print(pos_encoding.shape)
    
    plt.figure(figsize=(20,6))
    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()
    
test()