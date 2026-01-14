# Claude 开发规范和最佳实践

> **核心理念**: 预防胜于治疗,代码质量重于开发速度

## 🚨 必须遵守的开发流程

### 1. 开发完成后的强制检查清单

**每次完成代码后,必须执行以下步骤,缺一不可:**

- [ ] **在浏览器中实际运行代码** (不是脑内模拟,不是静态检查)
- [ ] **打开浏览器开发者工具 (F12)**
- [ ] **检查 Console 标签是否有 JavaScript 错误**
- [ ] **测试所有交互功能** (按钮、表单、动画等)
- [ ] **验证视觉效果是否符合预期**
- [ ] **测试边界情况和异常输入**
- [ ] **确保修复后重新测试** (不是改完就标记完成)

**禁止行为:**
- ❌ 编写完代码后不测试直接标记完成
- ❌ 只看代码不运行就认为"应该没问题"
- ❌ 修复一个问题后不重新测试整个功能
- ❌ 用户反馈问题后才去测试

---

## 📋 JavaScript 核心原则

### 原则 1: 变量作用域必须明确

**问题模式:**
```javascript
// ❌ 错误: 局部变量在函数外无法访问
function init() {
    const width = container.clientWidth;  // 局部变量
    const height = 500;
}

function draw() {
    console.log(width);  // ReferenceError: width is not defined
}
```

**通用规则:**
```javascript
// ✅ 正确: 明确哪些变量需要跨函数共享

// 1. 在文件开头集中声明所有全局变量
let width, height, svg;

// 2. 使用有意义的前缀避免命名冲突
let mainWidth, mainHeight;
let sidebarWidth, sidebarHeight;

// 3. 函数内只声明真正私有的变量
function init() {
    width = container.clientWidth;  // 赋值给全局变量
    height = 500;
    const temp = "只在init内使用";  // 真正的局部变量
}

function draw() {
    console.log(width);  // ✅ 可以访问
}
```

**检查清单:**
- [ ] 文件开头是否声明了所有需要共享的全局变量?
- [ ] 函数内的 `const`/`let` 变量是否只在函数内使用?
- [ ] 是否使用了命名前缀避免不同模块的变量冲突?
- [ ] 是否在 JSDoc 注释中标注了变量作用域?

---

### 原则 2: 异步操作和数据初始化的时机

**问题模式:**
```javascript
// ❌ 错误: 在数据为空时就初始化依赖数据的对象
function playAnimation() {
    const data = [];

    // 错误: simulation 在 data 为空时创建
    const simulation = d3.forceSimulation(data)
        .on("tick", update);

    function update() {
        // data 为空,这里会出错
        linkGroup.selectAll("line")
            .attr("x1", d => d.source.x);  // TypeError!
    }
}
```

**通用规则:**
```javascript
// ✅ 正确: 只在数据准备好后才创建依赖对象

function playAnimation() {
    const data = [];
    let simulation = null;  // 先声明为 null

    function update() {
        // 安全检查
        if (!simulation) return;

        linkGroup.selectAll("line")
            .attr("x1", d => d.source ? d.source.x : 0);
    }

    function addItem(item) {
        data.push(item);

        // ✅ 只在第一个数据添加后才创建 simulation
        if (!simulation && data.length > 0) {
            simulation = createSimulation(data);
            simulation.on("tick", update);
        } else if (simulation) {
            simulation.alpha(1).restart();  // 重启已存在的 simulation
        }
    }
}
```

**检查清单:**
- [ ] 是否在数据为空时创建了依赖该数据的对象?
- [ ] 是否使用了条件检查 (`if (!obj && data.length > 0)`)?
- [ ] 是否在数据更新后重启了 simulation/observer/监听器?
- [ ] 是否处理了数据未加载完成的情况?

---

### 原则 3: 数据绑定必须完整

**问题模式 (D3.js 特例,但适用于所有数据驱动框架):**
```javascript
// ❌ 错误: 创建 DOM 元素时没有绑定数据
const line = svg.append("line")
    .attr("x1", 0)
    .attr("y1", 0);

function update() {
    line.attr("x2", d => d.end.x);  // d 是 undefined
}
```

**通用规则 (适用于 React、Vue、D3 等):**
```javascript
// ✅ 正确: 创建元素时立即绑定数据

// D3.js 示例
const line = svg.append("line")
    .datum(dataItem)  // 必须绑定数据!
    .attr("x1", 0)
    .attr("y1", 0);

function update() {
    line.attr("x2", d => d.end.x);  // ✅ d 已绑定
}

// Vue 示例
<li v-for="item in items" :key="item.id">{{ item.name }}</li>

// React 示例
{items.map(item => <li key={item.id}>{item.name}</li>)}
```

**核心原则:**
> **任何动态内容都必须绑定到数据源,永远不要创建"孤儿元素"**

**检查清单:**
- [ ] 每个 DOM 元素是否都绑定了对应的数据?
- [ ] 是否使用了 `.datum()` (D3) 或 `v-for`/`map()` (Vue/React)?
- [ ] 是否在回调函数中检查了 `d` 或 `item` 是否存在?
- [ ] 是否为列表项设置了唯一的 `key` 属性?

---

### 原则 4: 重复初始化防护

**问题模式:**
```javascript
// ❌ 错误: 多次清空导致数据丢失
function reset() {
    container.innerHTML = "";  // 清空一次
    init();  // init() 内部又清空一次!
}

function init() {
    container.innerHTML = "";  // 重复清空
    // 创建内容...
}
```

**通用规则:**
```javascript
// ✅ 正确: 使用状态标志避免重复初始化

let isInitialized = false;

function init() {
    // 防御性检查: 只在第一次初始化
    if (isInitialized) {
        return;
    }

    container.innerHTML = "";
    // 创建内容...
    isInitialized = true;
}

function reset() {
    // 只重置数据,不重新初始化容器
    data = [];
    updateView();
}

// 或者: 使用条件检查
function init() {
    if (!container.children.length) {
        container.innerHTML = "";
        // 创建内容...
    }
}
```

**检查清单:**
- [ ] 是否有标志位 (`isInitialized`) 防止重复初始化?
- [ ] `init()` 和 `reset()` 的职责是否分离?
- [ ] 是否在函数开头检查了前置条件 (`if (!svg) return`)?
- [ ] 是否避免了"清空后立即又清空"的逻辑?

---

### 原则 5: 空值检查和防御性编程

**问题模式:**
```javascript
// ❌ 错误: 直接访问可能为 undefined 的属性
function update(items) {
    items.forEach(item => {
        console.log(item.name.toUpperCase());  // TypeError if item is undefined
    });
}
```

**通用规则:**
```javascript
// ✅ 正确: 每个访问点都检查是否存在

function update(items) {
    // 多层防御
    if (!items || !items.length) {
        return;  // 提前退出
    }

    items.forEach(item => {
        if (!item) return;  // 跳过无效项
        if (!item.name) return;  // 跳过没有 name 的项

        console.log(item.name.toUpperCase());
    });
}

// 或者使用可选链 (现代 JavaScript)
function update(items) {
    items?.forEach(item => {
        console.log(item?.name?.toUpperCase());
    });
}
```

**检查清单:**
- [ ] 访问对象属性前是否检查了对象是否存在?
- [ ] 数组方法 (`forEach`, `map`) 前是否检查了数组是否为空?
- [ ] 是否使用了可选链 (`?.`) 或空值合并 (`??`)?
- [ ] 是否在函数开头验证了输入参数?

---

## 🔧 通用编码规范

### 命名规范

**变量命名:**
```javascript
// ✅ 使用有意义的前缀区分不同作用域
let globalWidth, globalHeight;
let tempWidth, tempHeight;

// ✅ 使用模块前缀避免冲突
let erSvg, erWidth;      // ER 模块
let ecommerceSvg;        // 电商模块
let socialSvg;            // 社交模块

// ❌ 避免过于通用的名称
let width, height;        // 容易冲突
let data;                // 不明确
let temp;                // 无意义
```

**函数命名:**
```javascript
// ✅ 使用动词+名词的清晰命名
function initAnimation() { }
function resetData() { }
function updateView() { }
function handleError() { }

// ❌ 避免模糊的命名
function do() { }
function process() { }
function handle() { }
```

---

### 注释规范

**必须注释的情况:**
```javascript
// 1. 全局变量必须说明用途
/** @type {SVGElement} 主 SVG 容器 */
let mainSvg;

// 2. 复杂逻辑必须说明
// 注意: 这里使用 setTimeout 而不是 Promise,
// 因为需要与旧版 API 兼容
setTimeout(callback, 100);

// 3. 临时解决方案必须标记
// TODO: 重构为更高效的数据结构
const data = JSON.parse(jsonString);

// 4. 已知的坑必须标记
// FIXME: 在 Safari 浏览器中会闪烁,待修复
element.style.display = "none";
```

---

### 错误处理

**必须捕获错误的情况:**
```javascript
// ✅ 所有可能失败的操作都要 try-catch
function loadData(url) {
    try {
        const response = fetch(url);
        return response.json();
    } catch (error) {
        console.error("加载失败:", error);
        // 显示用户友好的错误信息
        showError("数据加载失败,请刷新页面重试");
        return null;  // 返回默认值
    }
}

// ✅ 使用 Promise 链处理异步错误
fetch(url)
    .then(response => response.json())
    .then(data => processData(data))
    .catch(error => {
        console.error("处理失败:", error);
        showError("数据处理失败");
    });
```

---

## 🎯 调试技巧

### 1. 使用 console.log 追踪数据流

```javascript
// ✅ 在关键点添加日志
function process(data) {
    console.log("输入数据:", data);
    const result = transform(data);
    console.log("转换结果:", result);
    return result;
}

// ✅ 使用分组日志
console.group("动画流程");
console.log("步骤1: 初始化", initResult);
console.log("步骤2: 加载数据", data);
console.log("步骤3: 渲染", renderResult);
console.groupEnd();
```

### 2. 使用断点调试

```javascript
// 在浏览器开发者工具中设置断点
function complexLogic(data) {
    debugger;  // 程序会在这里暂停
    // 然后可以逐步执行,检查变量值
    const result = data.map(item => item.value * 2);
    return result;
}
```

### 3. 使用断言验证假设

```javascript
// ✅ 在开发环境验证假设
function process(data) {
    console.assert(data !== null, "数据不应为 null");
    console.assert(data.length > 0, "数据不应为空");
    console.assert(typeof data[0].id === "number", "ID 必须是数字");

    // 继续处理...
}
```

---

## 🧪 测试策略

### 单元测试 (对核心函数)

```javascript
// ✅ 为关键函数编写测试用例
function testFormatName() {
    console.assert(formatName("john") === "John", "首字母大写失败");
    console.assert(formatName("") === "", "空字符串处理失败");
    console.assert(formatName(null) === "", "null 处理失败");
    console.log("✓ formatName 测试通过");
}

testFormatName();
```

### 集成测试 (对完整流程)

```javascript
// ✅ 测试完整用户流程
function testUserFlow() {
    // 1. 初始化
    init();
    console.assert(isInitialized, "初始化失败");

    // 2. 添加数据
    addItem({ id: 1, name: "测试" });
    console.assert(data.length === 1, "添加数据失败");

    // 3. 渲染
    render();
    console.assert(document.querySelectorAll(".item").length === 1, "渲染失败");

    // 4. 重置
    reset();
    console.assert(data.length === 0, "重置失败");

    console.log("✓ 用户流程测试通过");
}

testUserFlow();
```

---

## 🚀 性能优化原则

### 1. 避免重复计算

```javascript
// ❌ 错误: 每次循环都重新计算
for (let i = 0; i < items.length; i++) {
    const width = container.clientWidth;  // 重复计算
}

// ✅ 正确: 缓存计算结果
const width = container.clientWidth;
for (let i = 0; i < items.length; i++) {
    // 使用缓存的 width
}
```

### 2. 使用事件委托

```javascript
// ❌ 错误: 为每个元素绑定事件
items.forEach(item => {
    item.addEventListener("click", handleClick);
});

// ✅ 正确: 使用事件委托
container.addEventListener("click", (e) => {
    if (e.target.classList.contains("item")) {
        handleClick(e);
    }
});
```

### 3. 防抖和节流

```javascript
// ✅ 对高频事件使用防抖
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

const handleResize = debounce(() => {
    updateLayout();
}, 100);

window.addEventListener("resize", handleResize);
```

---

## 📖 参考资源

### 必读文档
- [MDN JavaScript Guide](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide)
- [D3.js API Reference](https://d3js.org/)
- [You Don't Know JS](https://github.com/getify/You-Dont-Know-JS)

### 工具推荐
- **ESLint**: 代码质量检查
- **Prettier**: 代码格式化
- **JSDoc**: 代码文档
- **Jest**: 单元测试框架

---

## 🎓 学习重点

### 必须掌握的 JavaScript 概念

1. **作用域和闭包**
   - 全局作用域 vs 函数作用域 vs 块级作用域
   - `var` vs `let` vs `const` 的区别
   - 闭包的工作原理和常见陷阱

2. **异步编程**
   - Callback vs Promise vs Async/Await
   - 事件循环 (Event Loop)
   - 错误处理模式

3. **数据驱动视图**
   - D3.js 的数据绑定机制
   - React 的状态管理
   - Vue 的响应式系统

4. **调试技巧**
   - 浏览器开发者工具的使用
   - 断点调试和日志追踪
   - 性能分析

---

## ✅ 最终检查清单

**提交代码前必须确认:**

- [ ] 在浏览器中实际运行并通过所有测试
- [ ] Console 无任何错误或警告
- [ ] 所有交互功能正常工作
- [ ] 代码已格式化 (Prettier)
- [ ] 代码通过 ESLint 检查
- [ ] 已添加必要的注释
- [ ] 已更新文档 (如果有 API 变更)
- [ ] 边界情况已处理 (空数据、错误输入等)
- [ ] 性能无明显问题 (无卡顿、无内存泄漏)
- [ ] 在多个浏览器中测试 (Chrome, Firefox, Safari)

---

## 🙏 承诺

从现在开始,我会:

1. ✅ **每次完成代码后立即在浏览器测试**
2. ✅ **打开开发者工具检查 Console**
3. ✅ **测试所有交互功能和边界情况**
4. ✅ **遵循通用编程原则,而不是依赖特定实现**
5. ✅ **使用防御性编程,添加空值检查**
6. ✅ **明确变量作用域,避免全局污染**
7. ✅ **确保无 bug 后再标记完成**
8. ✅ **记录所有发现的问题和解决方案**

---

**文档版本:** 2.0
**最后更新:** 2025-01-06
**核心理念**: 通用原则 > 特定实现,预防 > 修复
