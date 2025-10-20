# TransCox-Sparse R包封装建议

## 当前状态分析

### 现有R包结构
您的项目已经具备了完整的R包结构：
- ✅ `DESCRIPTION` 文件 - 包的基本信息
- ✅ `NAMESPACE` 文件 - 导出函数定义
- ✅ `R/` 目录 - R函数源码
- ✅ `inst/` 目录 - 安装文件（包含Python代码）
- ✅ `vignettes/` 目录 - 使用说明文档
- ✅ `tests/` 目录 - 测试文件
- ✅ `.Rbuildignore` 文件 - 构建忽略配置

### 稀疏功能集成状态
- ✅ 稀疏版本函数已添加到 `R/` 目录
- ✅ Python稀疏实现已放置在 `inst/python/` 目录
- ✅ 测试文件已创建并验证功能
- ✅ 文档已更新

## 封装建议

### 方案一：更新现有包（推荐）

**优势：**
- 保持版本连续性
- 用户升级简单
- 维护成本低

**实施步骤：**

#### 1. 更新DESCRIPTION文件
```r
Package: TransCox
Type: Package
Title: Transfer Learning for Cox Proportional Hazards Model with Sparse Regularization
Version: 0.2.0
Author: Ziyi Li <zli16@mdanderson.org>, [Your Name] <your.email@domain.com>
Maintainer: [Your Name] <your.email@domain.com>
Description: Enhanced Cox Proportional Hazards model with transfer learning capabilities. 
    Supports both standard and sparse regularization methods for high-dimensional survival analysis.
    Adaptively borrows information from external data sources with L1 regularization for feature selection.
License: GPL-2
Encoding: UTF-8
LazyData: true
Depends: R (>= 4.0), reticulate, survival
Imports: Matrix, glmnet
Suggests: knitr, rmarkdown, stats, testthat
VignetteBuilder: knitr
```

#### 2. 更新NAMESPACE文件
```r
import(reticulate)
import(survival)
importFrom("utils", "setTxtProgressBar", "txtProgressBar")
importFrom("Matrix", "Matrix")

export(
    GetAuxSurv,
    GetAuxSurv_Sparse,
    GetPrimaryParam,
    runTransCox_one,
    runTransCox_Sparse,
    runBtsp_transCox,
    SelLR_By_BIC,
    SelParam_By_BIC,
    SelParam_By_BIC_Sparse,
    GenSimData
)
```

#### 3. 创建包构建脚本
```r
# build_package.R
library(devtools)
library(roxygen2)

# 生成文档
document()

# 检查包
check()

# 构建包
build()

# 安装包
install()
```

### 方案二：创建新包

**优势：**
- 清晰的功能分离
- 独立的版本控制
- 专门针对稀疏功能优化

**包名建议：**
- `TransCoxSparse`
- `TransCox.Sparse`
- `SparseTransCox`

**实施步骤：**

#### 1. 创建新包结构
```bash
# 使用 usethis 包创建
library(usethis)
create_package("TransCoxSparse")
```

#### 2. 新包DESCRIPTION
```r
Package: TransCoxSparse
Type: Package
Title: Sparse Transfer Learning for Cox Proportional Hazards Model
Version: 1.0.0
Author: [Your Name] <your.email@domain.com>
Maintainer: [Your Name] <your.email@domain.com>
Description: Sparse regularization extension for TransCox method. 
    Implements L1 regularization for high-dimensional survival analysis with transfer learning.
License: GPL-2
Encoding: UTF-8
LazyData: true
Depends: R (>= 4.0), TransCox, reticulate, survival
Imports: Matrix, glmnet
Suggests: knitr, rmarkdown, stats, testthat
VignetteBuilder: knitr
```

## 推荐方案：更新现有包

### 理由：
1. **用户友好**：现有用户可以直接升级获得新功能
2. **维护简单**：只需维护一个包
3. **功能完整**：包含完整的TransCox功能集
4. **向后兼容**：保持原有API不变

### 具体实施计划：

#### 第一步：准备包文件
```bash
# 1. 更新版本信息
# 2. 添加新的导出函数
# 3. 更新依赖包列表
# 4. 创建函数文档
```

#### 第二步：创建函数文档
为每个新函数创建Roxygen2文档：

```r
#' Sparse Transfer Cox Regression
#'
#' @description 
#' Performs transfer learning for Cox regression with L1 regularization
#' for high-dimensional survival data.
#'
#' @param primData Primary dataset (data.frame)
#' @param auxData Auxiliary dataset (data.frame) 
#' @param cov Covariate names (character vector)
#' @param lambda_beta L1 regularization parameter (numeric)
#' @param ... Additional parameters
#'
#' @return List containing fitted model results
#' @export
#' @examples
#' \dontrun{
#' result <- runTransCox_Sparse(primData, auxData, cov, lambda_beta = 0.1)
#' }
runTransCox_Sparse <- function(...) {
    # 函数实现
}
```

#### 第三步：构建和测试
```r
# build_and_test.R
library(devtools)

# 1. 生成文档
document()

# 2. 运行测试
test()

# 3. 检查包
check()

# 4. 构建包
build()

# 5. 安装测试
install()
```

#### 第四步：发布准备
```r
# release_prep.R

# 1. 更新NEWS.md
# 2. 检查CRAN兼容性
check_rhub()
check_win_devel()

# 3. 提交到CRAN（可选）
release()
```

## 部署选项

### 选项1：GitHub发布
```bash
# 1. 推送到GitHub
git add .
git commit -m "Add sparse regularization features v0.2.0"
git tag v0.2.0
git push origin main --tags

# 2. 用户安装
devtools::install_github("yourusername/TransCox")
```

### 选项2：本地安装包
```bash
# 1. 构建tar.gz文件
R CMD build TransCox

# 2. 安装
R CMD INSTALL TransCox_0.2.0.tar.gz
```

### 选项3：私有仓库
```bash
# 1. 设置私有R包仓库
# 2. 上传包文件
# 3. 配置用户访问
```

## 质量保证

### 1. 测试覆盖
- ✅ 单元测试（已有）
- ✅ 集成测试（已有）
- ✅ 性能测试（已有）
- ⚠️ 需要添加：边界条件测试

### 2. 文档完整性
- ✅ 用户手册（已创建）
- ✅ 函数文档（需要Roxygen2格式）
- ✅ 示例代码（已有）
- ⚠️ 需要添加：变更日志

### 3. 兼容性检查
- ✅ R版本兼容性
- ✅ 依赖包版本
- ✅ 操作系统兼容性
- ⚠️ 需要测试：不同Python版本

## 维护计划

### 短期（1-3个月）
1. 完善文档和测试
2. 收集用户反馈
3. 修复发现的bug
4. 性能优化

### 中期（3-6个月）
1. 添加新的稀疏化方法
2. 支持更多数据格式
3. 提升计算效率
4. 扩展可视化功能

### 长期（6个月以上）
1. 集成到CRAN
2. 开发Web界面
3. 支持分布式计算
4. 机器学习集成

## 总结

**推荐采用方案一（更新现有包）**，具体原因：

1. **最小化用户迁移成本**
2. **保持代码库统一**
3. **简化维护工作**
4. **快速部署上线**

下一步行动：
1. 更新DESCRIPTION和NAMESPACE文件
2. 为新函数添加Roxygen2文档
3. 运行完整的包检查
4. 构建并测试安装
5. 推送到GitHub并创建release