import numpy as np
import numpy.typing as npt

def NE16_transform_weights(
        weight: npt.NDArray[np.uint8], bits: int, depthwise: bool = False
    ) -> npt.NDArray[np.uint8]:
    """Unroll weight into expected memory format

    Expected weight shape is (cout, cin, height, width).
    The output shape is: (cout, cinMajor, Bits, height x width, cinMinorBytes),
    where cinMajor is the ceil(cin / CIN_SUBTILE) and cinMinor has to be padded with 0 to CIN_SUBTILE.
    """
    # let's make the weights unsigned
    #print("Initial weigths",weight.tolist())
    weight = weight - 128
    if depthwise:
        weight = weight.transpose(1, 0, 2, 3)  # Swap cout and cin

    cout, cin, height, width = weight.shape
    #print(weight.shape)
    # Pad cin to be divisible with CIN_SUBTILE
    if cin % 16 != 0:
        cinPad = 16 - cin % 16
        weight = np.pad(
            weight,
            ((0, 0), (0, cinPad), (0, 0), (0, 0)),
            "constant",
            constant_values=0,
        )
        cin = cin + cinPad

    #print(weight)

    # Reshape into (cout, cinMajor, cinMinor, flattened spatial, 1)
    # The 1 at the end is required by the unpacking
    cinMajor = cin // 16
    cinMinor = 16
    weight = weight.reshape(cout, cinMajor, cinMinor, height * width, 1)

    #print("Pre unpack",weight.tolist())
    # Unpack 'bits' bits in little order, e.g. bits=4: 3 => [1, 1, 0, 0]
    # (cout, cinMajor, cinMinor, flattened spatial, Bits)
    weight = np.unpackbits(weight.astype(np.uint8), axis=-1, count=bits, bitorder="little")

    #print("After unpack",weight.tolist())
    # Shuffle bits so that the final shape is:
    # (cout, cinMajor, Bits, flattened spatial, cinMinor)
    weight = weight.transpose(0, 1, 4, 3, 2)

    # Prepare for packing
    # (cout, cinMajor, Bits, flattened spatial, cinMinorBytes, 8)
    cinMinorBytes = int(np.ceil(cinMinor / 8))
    weight = np.stack(np.split(weight, cinMinorBytes, axis=-1), axis=-2)

    # Pack
    # (cout, cinMajor, Bits, flattened spatial, cinMinorBytes)
    weight = np.packbits(weight, axis=-1, bitorder="little")
    #print(weight.shape)
    return weight.flatten()