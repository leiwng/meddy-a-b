import torch
import torch.nn.functional as F
from torchvision import transforms
from transformers import ViTForImageClassification
from PIL import Image
import os
import json
from source_code.api.config import (
    vit_model_class_map_path,
    vit_model_dir,
    vit_model_params_path,
)


class ImagePredictor:
    def __init__(self, model_path, class_names_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载类别名称
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, "r") as f:
                self.class_names = json.load(f)
        else:
            print("警告：未提供类别名称文件，将使用数字索引作为类别标签。")
            self.class_names = None

        # 图像预处理
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, model_path):
        # 获取类别数量
        if self.class_names:
            num_classes = len(self.class_names)
        else:
            # 从模型文件中获取类别数量
            state_dict = torch.load(model_path, map_location=self.device)
            classifier_weight = state_dict.get("classifier.weight", None)
            if classifier_weight is not None:
                num_classes = classifier_weight.size(0)
            else:
                raise ValueError(
                    "无法确定类别数量，请提供类别名称文件或确保模型文件包含分类器权重信息。"
                )

        # 初始化模型
        model = ViTForImageClassification.from_pretrained(
            vit_model_dir,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        # 加载训练好的权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        return model

    def predict_image(self, image_path, return_probs=False):
        # 加载并预处理图像
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # 进行预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)

            # 获取预测结果
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0][pred_class].item()

            # 获取类别名称
            if self.class_names:
                pred_class_name = self.class_names[str(pred_class)]
            else:
                pred_class_name = str(pred_class)

            if return_probs:
                # 返回所有类别的概率
                all_probs = probs[0].cpu().numpy()
                class_probs = {}
                for i, prob in enumerate(all_probs):
                    class_name = (
                        self.class_names[str(i)] if self.class_names else str(i)
                    )
                    class_probs[class_name] = float(prob)
                return pred_class_name, pred_prob, class_probs
            else:
                return pred_class_name, pred_prob


# 初始化预测器
predictor = ImagePredictor(vit_model_params_path, vit_model_class_map_path)

if __name__ == "__main__":
    result = predictor.predict_image("/data/cjl/project/resnet_chromo_cls/data/val/crop/L2505100034.001.21_0.034.png")
    print(result)
