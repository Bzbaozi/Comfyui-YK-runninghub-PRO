# 🚀 YK-影客AI-RUNHUB全能图片PRO

> **ComfyUI 自定义节点**｜支持 RunningHub 社区版 / 官方PRO版 / 混合智能模式  
> 批量图生图 · 多参考图 · 自动失败跳过 · 多图床支持

在 ComfyUI 节点菜单中显示为：  
**YK-影客AI-RUNHUB全能图片PRO（社区/官方/混合）**

---

## 📦 安装步骤

```bash
# 1. 进入 ComfyUI 的 custom_nodes 目录
cd ComfyUI/custom_nodes

# 2. 克隆本项目
git clone https://github.com/Bzbaozi/Comfyui-YK-runninghub-PRO.git

# 3. （可选）如需使用阿里云 OSS，请安装依赖
pip install oss2

# 4. 重启 ComfyUI
📸 图床配置指南（必读）
本节点需将参考图上传至公网图床，RunningHub 才能访问。请选择一种方式：
🔹 方式一：ImgBB（推荐新手，免费）
访问 https://imgbb.com/ 注册账号
登录后进入 API 页面 获取 API Key
在节点参数中：
image_hosting → 选择 ImgBB
imgbb_api_key → 填入你的 API Key
✅ 优点：免费、无需配置、5MB/图
⚠️ 注意：免费账户有速率限制，高频使用可能触发限流
🔹 方式二：阿里云 OSS（推荐高频/商业使用）
开通 OSS 服务
进入 阿里云 OSS 控制台
创建 Bucket（建议权限设为 公共读）
获取 AccessKey
进入 RAM 访问控制台
创建子用户 → 授予 AliyunOSSFullAccess 权限
获取 AccessKey ID 和 AccessKey Secret
填写节点参数
image_hosting → 选择 阿里云 OSS
oss_access_key_id → 填入 AccessKey ID
oss_access_key_secret → 填入 AccessKey Secret
oss_bucket_name → 你的 Bucket 名称（如 my-comfy-bucket）
oss_endpoint → 区域 Endpoint（如 oss-cn-beijing.aliyuncs.com）
🔗 查看所有区域 Endpoint
✅ 优点：高速稳定、无公开限流
⚠️ 注意：会产生少量存储/流量费用（约 ¥0.1/GB）
🔐 安全警告：切勿将 AccessKey 提交到 Git 或分享给他人！
🔑 RunningHub API 密钥
访问 https://www.runninghub.cn/
登录后进入「个人中心」→「API 密钥」
复制密钥，填入节点的 runninghub_api_key 字段
表格
版本	免费额度	最高分辨率	并发数
社区版	有	4K	低
官方PRO版	需订阅	8K	高
🎯 使用示例
连接 1～3 张参考图到 image_A_a, image_A_b...
输入提示词到 prompt_1
设置 batch_count_1 = 4（生成 4 个变体）
选择图床并填写对应密钥
点击执行 → 输出包含 所有成功图像（失败组自动跳过）
❓ 常见问题
Q：输出是 64x64 黑图？
A：表示该组全部失败。请检查：
RunningHub API 密钥是否有效
图床配置是否正确
参考图是否上传成功（查看日志）
Q：社区版能用 8K 吗？
A：不能。8K 会自动降级为 4K。
Q：必须传参考图吗？
A：是的。本节点为 图生图（Image-to-Image），至少需 1 张参考图。
Q：支持中文提示词吗？
A：支持！RunningHub 官方已适配中文。
🌟 由 Bzbaozi 开发｜版本：1.0
⭐ 如果你觉得好用，欢迎 Star 本项目！
