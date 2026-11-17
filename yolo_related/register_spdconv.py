"""SPDConv注册模块：将SPDConv注入ultralytics，使其可在YAML中被识别"""
import sys
import os

# 确保my_modules可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def register():
    """将SPDConv、C2fAttn、C2fDARM注册到ultralytics模块中"""
    from my_modules import (
        SPDConv,
        C2fAttn as _C2fAttnCustom,
        C2fDARM as _C2fDARMCustom,
        C2fGDARM as _C2fGDARMCustom,
        IlluminationResidualGate,
        SimAMBlock,
    )
    import ultralytics.nn.modules.conv as ultralytics_conv
    import ultralytics.nn.modules.block as ultralytics_block
    import ultralytics.nn.modules as mods
    import ultralytics.nn.tasks as tasks

    if not hasattr(ultralytics_conv, 'SPDConv'):
        setattr(ultralytics_conv, 'SPDConv', SPDConv)
        if 'SPDConv' not in mods.__dict__:
            mods.__dict__['SPDConv'] = SPDConv
        print(f"[register] SPDConv 已注册")

    setattr(ultralytics_block, 'C2fAttn', _C2fAttnCustom)
    if 'C2fAttn' not in mods.__dict__:
        mods.__dict__['C2fAttn'] = _C2fAttnCustom
    tasks.C2fAttn = _C2fAttnCustom
    print(f"[register] C2fAttn 已注册")

    setattr(ultralytics_block, 'C2fDARM', _C2fDARMCustom)
    if 'C2fDARM' not in mods.__dict__:
        mods.__dict__['C2fDARM'] = _C2fDARMCustom
    tasks.C2fDARM = _C2fDARMCustom

    # Patch parse_model to recognize C2fDARM as a C2f-class module
    # Without this, ultralytics doesn't inject n properly for C2fDARM
    import ultralytics.nn.tasks as ult_tasks
    _original_parse_model = ult_tasks.parse_model

    def _patched_parse_model(d, ch, verbose=True):
        # Temporarily add C2fDARM to the C2f set so parse_model handles it correctly
        import types
        # We need to insert C2fDARM into the C2f-class recognition
        # The simplest way: make isinstance check pass
        # Actually: patch the code path where "m in {C2f, ...}" is checked
        return _original_parse_model(d, ch, verbose)

    setattr(ultralytics_block, 'C2fGDARM', _C2fGDARMCustom)
    if 'C2fGDARM' not in mods.__dict__:
        mods.__dict__['C2fGDARM'] = _C2fGDARMCustom
    tasks.C2fGDARM = _C2fGDARMCustom
    print(f"[register] C2fDARM 已注册（原创模块：GAP+GMP+GVP 多统计量注意力）")
    print(f"[register] C2fGDARM 已注册（原创模块：GAP+GMP+GVP + 自适应门控）")

    # SimAM — parameter-free attention
    setattr(ultralytics_block, 'SimAMBlock', SimAMBlock)
    if 'SimAMBlock' not in mods.__dict__:
        mods.__dict__['SimAMBlock'] = SimAMBlock
    tasks.SimAMBlock = SimAMBlock  # Also register in tasks namespace for parse_model
    print(f"[register] SimAMBlock 已注册（无参数3D注意力）")

    # Standalone shape-preserving block: parse_model leaves the channel count
    # unchanged and forwards the explicitly declared channel argument.
    setattr(ultralytics_block, 'IlluminationResidualGate', IlluminationResidualGate)
    if 'IlluminationResidualGate' not in mods.__dict__:
        mods.__dict__['IlluminationResidualGate'] = IlluminationResidualGate
    tasks.IlluminationResidualGate = IlluminationResidualGate
    print(f"[register] IlluminationResidualGate 已注册（Model H 光照感知残差门控）")


def load_yolo_with_spdconv(weights_path):
    """加载YOLO模型（自动注册SPDConv）"""
    register()
    from ultralytics import YOLO
    return YOLO(weights_path)


def create_yolo_from_yaml(yaml_path):
    """从YAML创建YOLO模型（自动注册SPDConv）"""
    register()
    from ultralytics import YOLO
    return YOLO(yaml_path)
