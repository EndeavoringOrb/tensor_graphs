"""Public native ABI and JSON configuration descriptions."""
import math
import struct
import sys

DTYPES = dict(enumerate(('FLOAT32', 'INT32', 'INT64', 'BF16', 'BOOL', 'ANY', 'INT8',
                         'E2M1_PACKED_INT8', 'E2M1', 'F8_E8M0', 'F8_E4M3')))
CONTRACT = {
    'constant_byte_order': sys.byteorder,
    'match': 'bool match(const std::vector<TensorNode> &inputs, const TensorNode &output)',
    'run': 'void run(const KernelContext &ctx)',
    'reference_factory': 'LogicalId referenceFactory(const std::vector<LogicalId> &inputs, Graph &graph)',
    'context': {'inputs': 'std::vector<const void *>', 'outputs': 'std::vector<void *>',
                'inViews/outViews': 'std::vector<TensorView>', 'shape': 'const std::vector<uint32_t> &getShape() const',
                'strides': 'std::vector<uint64_t>, in elements', 'cuda_stream': 'void *cuda_stream(size_t idx = 0) const'},
    'registration': ['name', 'min_num_inputs', 'max_num_inputs', 'match', 'run', 'reference_factory',
                     'safe_inplace_idxs', 'output_mem_space', 'engines', 'dtypes', 'dummy_shapes',
                     'contiguous', 'input_mem_spaces', 'is_view', 'isReference', 'inferView'],
    'registration_parameters': {
        'name': 'std::string: literal registration name, unique against the goal baseline',
        'min_num_inputs/max_num_inputs': 'uint32_t: inclusive accepted input-count bounds',
        'match': 'MatchFunc: optional additional configuration predicate; nullptr accepts built-in checks',
        'run': 'KernelFunc: executable implementation',
        'reference_factory': 'ReferenceFactory: constructs the logical graph used to identify the target and test correctness',
        'safe_inplace_idxs': 'std::vector<uint32_t>: input indices whose buffers may alias the output safely',
        'output_mem_space': 'MemSpace: output storage type and topology role',
        'engines': 'std::vector<Engine>: execution resources and topology roles occupied by the kernel',
        'dtypes': 'std::vector<DType>: required input dtypes',
        'dummy_shapes': 'std::vector<std::vector<uint32_t>>: input shapes used to construct the reference graph',
        'contiguous': 'std::vector<bool>: true requires the corresponding input to be contiguous',
        'input_mem_spaces': 'std::vector<MemSpace>: input storage types and topology roles',
        'is_view': 'bool: view registration flag; executable candidates keep false',
        'isReference': 'bool: baseline reference registration flag; candidates keep false',
        'inferView': 'InferViewFunc: view metadata callback; executable candidates keep nullptr'},
    'registration_defaults': {'safe_inplace_idxs': 'required (use {})', 'output_mem_space': 'required',
        'engines': 'required', 'dtypes': '{} (must supply min_num_inputs entries)',
        'dummy_shapes': '{} (supply one shape per input for reference discovery)',
        'contiguous': '{} (must supply min_num_inputs entries)',
        'input_mem_spaces': '{} (must supply min_num_inputs entries)',
        'is_view': 'false', 'isReference': 'false', 'inferView': 'nullptr'},
    'registration_notes': 'safe_inplace_idxs lists inputs whose buffers may safely alias output. '
        'dtypes, contiguous and input_mem_spaces must each have min_num_inputs entries; '
        'the final entry repeats for additional inputs. dummy_shapes instantiate the reference graph. '
        'Memory spaces and engines use hardware topology roles. Executable candidates use the defaults '
        'is_view=false, isReference=false and inferView=nullptr. See core/kernels.hpp and core/types.hpp. '
        'Names must differ from this goal snapshot baseline; independent jobs may reuse candidate names.',
    'configuration': {'input_shapes/output_shape': 'Dimensions (uint32_t)',
                      'input_strides/output_strides': 'Strides in elements (uint64_t), not bytes',
                      'input_dtypes/output_dtype': DTYPES,
                      'input_constants': 'Per-input raw native-endian bytes; [] means no captured constant. '
                          'Decoded values supplement, and never replace, raw bytes.',
                      'decoded_input_constants': 'Per-input arrays for FLOAT32, INT32, INT64, BF16, BOOL and INT8; '
                          'null for absent constants or unsupported codecs. Non-finite floats are strings.'},
    'matching': 'Matching kernels are eligible implementations, not measured winners. '
        'The harness resolves candidate topology without requiring original workload memory/engine assignments.'}


def decodeConstants(configuration):
    decoded = []
    formats = {0: 'f', 1: 'i', 2: 'q', 4: '?', 6: 'b'}
    for dtype, raw in zip(configuration['input_dtypes'], configuration['input_constants'], strict=True):
        data = bytes(raw)
        fmt = formats.get(dtype)
        if not data or (fmt is None and dtype != 3):
            decoded.append(None)
            continue
        size = 2 if dtype == 3 else struct.calcsize('=' + fmt)
        if len(data) % size:
            raise ValueError('Constant byte count does not match dtype size')
        values = ([struct.unpack('=f', struct.pack('=I', value[0] << 16))[0]
                   for value in struct.iter_unpack('=H', data)] if dtype == 3 else
                  [value[0] for value in struct.iter_unpack('=' + fmt, data)])
        decoded.append([value if not isinstance(value, float) or math.isfinite(value) else str(value)
                        for value in values])
    return decoded


def targetKernels(metadata, expression):
    return list({str(kernel['kernel_id']): kernel for kernel in metadata['kernels']
                 if kernel['reference_expression'] == expression}.values())
