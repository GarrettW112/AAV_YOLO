import torch

def check_torch_cuda():
    print(f"PyTorch version: {torch.__version__}")
    
    is_available = torch.cuda.is_available()
    print(f"CUDA available:  {is_available}")

    if not is_available:
        print("\n PyTorch cannot detect your GPU.")
        return

    print("--- GPU Details ---")
    try:
        print(f"CUDA version:    {torch.version.cuda}")
        count = torch.cuda.device_count()
        print(f"Device count:    {count}")
        
        if count > 0:
            current_device = torch.cuda.current_device()
            print(f"Current device:  {current_device}")
            print(f"Device name:     {torch.cuda.get_device_name(current_device)}")
    except Exception as e:
        print(f"Error getting GPU details: {e}")

if __name__ == "__main__":
    check_torch_cuda()