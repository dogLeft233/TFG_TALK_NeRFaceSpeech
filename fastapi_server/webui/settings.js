/**
 * 设置管理模块 - 使用后端API与数据库交互
 * 所有页面共享此模块来管理设置
 */

const SETTINGS_API_BASE = "http://localhost:8000/api/settings";

// 默认设置（仅在数据库中没有设置时使用）
const DEFAULT_SETTINGS = {
    nerf_theme: "tech",
    nerf_font: "Inter",
    nerf_font_size: "medium",
    nerf_custom_font_size: "14"
};

/**
 * 从后端获取所有设置
 */
async function fetchSettings() {
    try {
        const response = await fetch(SETTINGS_API_BASE);
        if (!response.ok) {
            console.error(`获取设置失败: HTTP ${response.status}`);
            return DEFAULT_SETTINGS;
        }
        const result = await response.json();
        if (result.success && result.data) {
            return result.data;
        } else {
            console.warn("API返回格式异常，使用默认设置");
            return DEFAULT_SETTINGS;
        }
    } catch (error) {
        console.error("获取设置失败:", error);
        return DEFAULT_SETTINGS;
    }
}

/**
 * 获取单个设置值（从数据库读取）
 */
async function getSetting(key) {
    const settings = await fetchSettings();
    return settings[key] || DEFAULT_SETTINGS[key];
}

/**
 * 更新单个设置（直接写入数据库，不使用缓存）
 */
async function updateSetting(key, value) {
    try {
        // 后端API期望 embed=True，即 {"value": "xxx"} 格式
        const response = await fetch(`${SETTINGS_API_BASE}/${key}`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({"value": String(value)})  // 发送 {"value": "xxx"} 格式
        });
        const result = await response.json();
        return result.success || false;
    } catch (error) {
        console.error("更新设置失败:", error);
        return false;
    }
}

/**
 * 批量更新设置（直接写入数据库，不使用缓存）
 */
async function updateSettings(settings) {
    try {
        const response = await fetch(SETTINGS_API_BASE, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(settings)
        });
        const result = await response.json();
        return result.success || false;
    } catch (error) {
        console.error("批量更新设置失败:", error);
        return false;
    }
}

/**
 * 应用主题到页面
 */
function applyThemeToPage(theme) {
    if (document.body) {
        document.body.className = 'theme-' + theme;
    } else {
        // 如果body还没加载，先设置到documentElement，等body加载后再应用
        document.documentElement.setAttribute('data-pending-theme', theme);
        // 监听DOMContentLoaded来应用主题
        if (document.readyState === 'loading') {
            const applyThemeHandler = () => {
                if (document.body) {
                    document.body.className = 'theme-' + theme;
                    document.documentElement.removeAttribute('data-pending-theme');
                }
                document.removeEventListener('DOMContentLoaded', applyThemeHandler);
            };
            document.addEventListener('DOMContentLoaded', applyThemeHandler);
        }
    }
}

/**
 * 应用字体到页面
 */
function applyFontToPage(font) {
    document.body.style.fontFamily = font;
}

/**
 * 应用字体大小到页面
 */
function applyFontSizeToPage(size, customSize = null) {
    let fontSize;
    if (size === 'small') fontSize = '12px';
    else if (size === 'medium') fontSize = '14px';
    else if (size === 'large') fontSize = '16px';
    else if (size === 'custom') {
        fontSize = (customSize || DEFAULT_SETTINGS.nerf_custom_font_size || '14') + 'px';
    }
    if (fontSize) {
        document.documentElement.style.setProperty('--base-font-size', fontSize);
        document.documentElement.style.fontSize = fontSize;
    }
}

/**
 * 应用主题（会保存到数据库）
 */
async function applyTheme(theme) {
    applyThemeToPage(theme);
    await updateSetting('nerf_theme', theme);
    // 更新选择状态（延迟执行，确保DOM已加载）
    if (document.readyState === 'complete') {
        await updateThemeSelection();
    } else {
        document.addEventListener('DOMContentLoaded', async () => {
            await updateThemeSelection();
        }, { once: true });
    }
}

/**
 * 应用字体（会保存到数据库）
 */
async function applyFont(font) {
    applyFontToPage(font);
    await updateSetting('nerf_font', font);
    // 更新选择状态（延迟执行，确保DOM已加载）
    if (document.readyState === 'complete') {
        await updateFontSelection();
    } else {
        document.addEventListener('DOMContentLoaded', async () => {
            await updateFontSelection();
        }, { once: true });
    }
}

/**
 * 应用字体大小（会保存到数据库）
 */
async function applyFontSize(size) {
    let fontSize;
    let customValue = null;
    
    if (size === 'small') fontSize = '12px';
    else if (size === 'medium') fontSize = '14px';
    else if (size === 'large') fontSize = '16px';
    else if (size === 'custom') {
        const customInput = document.getElementById('customSizeValue');
        if (customInput && customInput.value) {
            customValue = customInput.value;
            fontSize = customValue + 'px';
            await updateSetting('nerf_custom_font_size', customValue);
        } else {
            return; // 如果没有自定义值，不应用
        }
    }
    
    if (fontSize) {
        applyFontSizeToPage(size, customValue);
        await updateSetting('nerf_font_size', size);
        // 更新选择状态（延迟执行，确保DOM已加载）
        if (document.readyState === 'complete') {
            await updateFontSizeSelection();
        } else {
            document.addEventListener('DOMContentLoaded', async () => {
                await updateFontSizeSelection();
            }, { once: true });
        }
    }
}

/**
 * 更新主题选择状态（从数据库读取当前设置）
 */
async function updateThemeSelection() {
    if (!document.body) return; // 如果body还没加载，直接返回
    try {
        const settings = await fetchSettings();
        const currentTheme = settings.nerf_theme || DEFAULT_SETTINGS.nerf_theme;
        const themeCards = document.querySelectorAll('[data-theme]');
        if (themeCards.length === 0) return; // 如果没有找到元素，直接返回
        themeCards.forEach(card => {
            card.classList.toggle('selected', card.dataset.theme === currentTheme);
        });
    } catch (error) {
        console.error("更新主题选择状态失败:", error);
    }
}

/**
 * 更新字体选择状态（从数据库读取当前设置）
 */
async function updateFontSelection() {
    if (!document.body) return; // 如果body还没加载，直接返回
    try {
        const settings = await fetchSettings();
        const currentFont = settings.nerf_font || DEFAULT_SETTINGS.nerf_font;
        const fontCards = document.querySelectorAll('[data-font]');
        if (fontCards.length === 0) return; // 如果没有找到元素，直接返回
        fontCards.forEach(card => {
            card.classList.toggle('selected', card.dataset.font === currentFont);
        });
    } catch (error) {
        console.error("更新字体选择状态失败:", error);
    }
}

/**
 * 更新字号选择状态（从数据库读取当前设置）
 */
async function updateFontSizeSelection() {
    if (!document.body) return; // 如果body还没加载，直接返回
    try {
        const settings = await fetchSettings();
        const currentSize = settings.nerf_font_size || DEFAULT_SETTINGS.nerf_font_size;
        const sizeButtons = document.querySelectorAll('[data-size]');
        if (sizeButtons.length > 0) {
            sizeButtons.forEach(btn => {
                btn.classList.toggle('selected', btn.dataset.size === currentSize);
            });
        }
        
        const customInput = document.getElementById('customSizeInput');
        if (customInput) {
            if (currentSize === 'custom') {
                customInput.style.display = 'block';
                const customValue = settings.nerf_custom_font_size || DEFAULT_SETTINGS.nerf_custom_font_size;
                const customValueInput = document.getElementById('customSizeValue');
                if (customValueInput) {
                    customValueInput.value = customValue;
                }
            } else {
                customInput.style.display = 'none';
            }
        }
    } catch (error) {
        console.error("更新字号选择状态失败:", error);
    }
}

/**
 * 从数据库读取设置并应用（用于除start.html外的所有页面）
 * 总是从数据库读取，不使用缓存
 */
async function applySettingsOnly() {
    try {
        // 总是从数据库读取设置
        const settings = await fetchSettings();
        
        // 应用数据库中的设置
        const theme = settings.nerf_theme || DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);
        
        // 应用字体
        const font = settings.nerf_font || DEFAULT_SETTINGS.nerf_font;
        if (document.body) {
            applyFontToPage(font);
        } else {
            document.documentElement.style.fontFamily = font;
        }
        
        // 应用字体大小
        const fontSize = settings.nerf_font_size || DEFAULT_SETTINGS.nerf_font_size;
        const customSize = settings.nerf_custom_font_size || DEFAULT_SETTINGS.nerf_custom_font_size;
        applyFontSizeToPage(fontSize, fontSize === 'custom' ? customSize : null);
    } catch (error) {
        console.error("从数据库读取设置失败:", error);
        // 如果读取失败，使用默认设置
        const theme = DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);
        if (document.body) {
            applyFontToPage(DEFAULT_SETTINGS.nerf_font);
        } else {
            document.documentElement.style.fontFamily = DEFAULT_SETTINGS.nerf_font;
        }
        applyFontSizeToPage(DEFAULT_SETTINGS.nerf_font_size, DEFAULT_SETTINGS.nerf_font_size === 'custom' ? DEFAULT_SETTINGS.nerf_custom_font_size : null);
    }
}

/**
 * 初始化设置（页面加载时调用）
 * 这个函数会从数据库获取设置并应用
 * 注意：只在开发者页面（start.html）的DOMContentLoaded中调用，其他页面使用 applySettingsOnly()
 */
async function initSettings() {
    try {
        // 获取所有设置（从数据库）
        const settings = await fetchSettings();
        
        // 使用数据库中的设置，如果数据库中没有则使用默认值（但不更新数据库）
        const theme = settings.nerf_theme || DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);
        
        // 应用字体
        const font = settings.nerf_font || DEFAULT_SETTINGS.nerf_font;
        if (document.body) {
            applyFontToPage(font);
        } else {
            document.documentElement.style.fontFamily = font;
        }
        
        // 应用字体大小
        const fontSize = settings.nerf_font_size || DEFAULT_SETTINGS.nerf_font_size;
        const customSize = settings.nerf_custom_font_size || DEFAULT_SETTINGS.nerf_custom_font_size;
        applyFontSizeToPage(fontSize, fontSize === 'custom' ? customSize : null);
        
        // 更新选择状态（此时DOM应该已经加载完成）
        await updateThemeSelection();
        await updateFontSelection();
        await updateFontSizeSelection();
        
        return settings;
    } catch (error) {
        console.error("初始化设置失败:", error);
        // 如果获取设置失败，使用默认设置（仅用于显示，不保存到数据库）
        const theme = DEFAULT_SETTINGS.nerf_theme;
        applyThemeToPage(theme);
        // 即使失败也尝试更新UI
        try {
            await updateThemeSelection();
            await updateFontSelection();
            await updateFontSizeSelection();
        } catch (e) {
            // 忽略UI更新错误
        }
        return DEFAULT_SETTINGS;
    }
}

// 注意：不在脚本加载时自动执行初始化
// start.html 应该在 DOMContentLoaded 时手动调用 initSettings()
// 其他页面应该调用 applySettingsOnly() 来应用缓存的设置
// 这样可以避免阻塞页面加载，即使 fetchSettings() 失败也不会影响页面渲染

