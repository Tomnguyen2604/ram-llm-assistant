import torch
print('torch.__version__:', getattr(torch, '__version__', None))
print('cuda_available:', torch.cuda.is_available())
print('cuda_device_count:', torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print('device_name:', torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print('total_memory_mb:', int(props.total_memory/1024**2))
    except Exception as e:
        print('device_info_error:', e)
