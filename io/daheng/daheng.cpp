#include "daheng.hpp"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <vector>
#include <mutex>
#include <iostream>
#include <cstring> 

#include "include/GxIAPILegacy.h"
#include "include/DxImageProc.h"

constexpr double MS_TO_US = 1000.0;
// 保持队列长度为1，确保低延迟
constexpr size_t THREAD_SAFE_QUEUE_MAX_SIZE = 1;

namespace io
{

static DaHeng* g_daheng_instance = nullptr;

static const char* gx_status_to_string(GX_STATUS status)
{
    switch (status) {
    case GX_STATUS_SUCCESS: return "SUCCESS";
    default: return "ERROR"; 
    }
}

void GX_STDC OnFrameCallback(GX_FRAME_CALLBACK_PARAM* pFrame)
{
    if (g_daheng_instance && pFrame && pFrame->status == GX_FRAME_STATUS_SUCCESS)
        g_daheng_instance->onFrameCallback(pFrame);
}

// -------------------- 图像回调 (带自动降级容错) --------------------
void DaHeng::onFrameCallback(GX_FRAME_CALLBACK_PARAM* pFrame)
{
    if (!pFrame || pFrame->status != GX_FRAME_STATUS_SUCCESS)
        return;

    CameraData data;
    cv::Mat bgr(pFrame->nHeight, pFrame->nWidth, CV_8UC3);
    bool process_success = false;

    // -------------------- 方案 A: 大恒 SDK 一体化加速 (转码+Gamma) --------------------
    if (!lut_content_.empty()) 
    {
        COLOR_IMG_PROCESS img_proc_config;
        memset(&img_proc_config, 0, sizeof(img_proc_config));
        
        img_proc_config.bDefectivePixelCorrect = false;
        img_proc_config.bDenoise = false;
        img_proc_config.bSharpness = false;
        img_proc_config.bAccelerate = true; 
        img_proc_config.parrCC = nullptr;
        
        img_proc_config.pProLut = lut_content_.data();
        img_proc_config.nLutLength = (VxUint16)lut_content_.size();
        
        img_proc_config.cvType = RAW2RGB_NEIGHBOUR; 
        img_proc_config.emLayOut = (DX_PIXEL_COLOR_FILTER)pixel_color_filter_;
        img_proc_config.bFlip = false;

        // 尝试执行
        VxInt32 status = DxRaw8ImgProcess(
            (void*)pFrame->pImgBuf, 
            (void*)bgr.data, 
            pFrame->nWidth, 
            pFrame->nHeight, 
            &img_proc_config
        );

        if (status == DX_OK) {
            process_success = true;
        } else {
            // 打印一次错误信息 (防止刷屏)
            static bool error_printed = false;
            if (!error_printed) {
                std::cerr << "[DaHeng] Error: DxRaw8ImgProcess failed! Code: " << status 
                          << ". Switching to Software fallback." << std::endl;
                error_printed = true;
            }
        }
    }

    // -------------------- 方案 B: 软件降级 (OpenCV转码 + cv::LUT) --------------------
    // 如果方案A失败，或者LUT未生成，执行此方案保证程序不卡死
    if (!process_success) 
    {
        DX_BAYER_CONVERT_TYPE convert_type = RAW2RGB_NEIGHBOUR;
        DX_PIXEL_COLOR_FILTER color_filter = (DX_PIXEL_COLOR_FILTER)pixel_color_filter_;
        DX_RGB_CHANNEL_ORDER order = DX_ORDER_BGR;

        // 1. 转码
        if (DxRaw8toRGB24Ex((void*)pFrame->pImgBuf, (void*)bgr.data,
                            pFrame->nWidth, pFrame->nHeight,
                            convert_type, color_filter, false, order) != DX_OK)
            return; // 真的失败了

        // 2. 软件 Gamma (利用 cv::Mat 版本的 LUT)
        // 我们利用 vector 生成一个 cv::Mat LUT
        if (!lut_content_.empty()) {
            cv::Mat cv_lut(1, 256, CV_8U, lut_content_.data());
            cv::LUT(bgr, cv_lut, bgr);
        }
    }
    data.img = std::move(bgr);
    data.timestamp = std::chrono::steady_clock::now();

    queue_.push(std::move(data));
    frame_count_++;
}

// -------------------- 构造函数 --------------------
DaHeng::DaHeng(double exposure_ms, double gain, const std::string& vid_pid)
    : exposure_us_(exposure_ms * MS_TO_US),
      gain_(gain),
      handle_(nullptr),
      is_color_camera_(true),
      capturing_(false),
      capture_quit_(false),
      queue_(THREAD_SAFE_QUEUE_MAX_SIZE),
      pixel_color_filter_(0),
      frame_count_(0),
      last_fps_time_(std::chrono::steady_clock::now())
{
    GX_STATUS status;
    g_daheng_instance = this;

    if ((status = GXInitLib()) != GX_STATUS_SUCCESS) throw std::runtime_error("GXInitLib failed");

    uint32_t num = 0;
    GXUpdateDeviceList(&num, 1000);
    if (num == 0) throw std::runtime_error("No Daheng camera found");

    GX_OPEN_PARAM open;
    open.accessMode = GX_ACCESS_EXCLUSIVE;
    open.openMode = GX_OPEN_INDEX;
    open.pszContent = (char*)"1";

    if ((status = GXOpenDevice(&open, &handle_)) != GX_STATUS_SUCCESS) throw std::runtime_error("GXOpenDevice failed");

    // 1. 生成 Gamma 2.2 LUT
    lut_content_.resize(256);
    VxUint16 lut_len = 256;
    if (DxGetLut(0, 1.5, 0, lut_content_.data(), &lut_len) == DX_OK) {
        std::cout << "[DaHeng] Software Gamma LUT (2.2) generated successfully." << std::endl;
    } else {
        std::cerr << "[DaHeng] Failed to generate Gamma LUT!" << std::endl;
        lut_content_.clear(); // 清空以触发 fallback 的无LUT模式
    }

    GXGetEnum(handle_, GX_ENUM_PIXEL_COLOR_FILTER, &pixel_color_filter_);

    GXSetEnum(handle_, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_OFF);
    GXSetEnum(handle_, GX_ENUM_ACQUISITION_MODE, GX_ACQ_MODE_CONTINUOUS);
    GXSetEnum(handle_, GX_ENUM_EXPOSURE_MODE, GX_EXPOSURE_MODE_TIMED);
    GXSetEnum(handle_, GX_ENUM_EXPOSURE_AUTO, GX_EXPOSURE_AUTO_OFF);
    GXSetFloat(handle_, GX_FLOAT_EXPOSURE_TIME, exposure_us_);
    GXSetEnum(handle_, GX_ENUM_GAIN_AUTO, GX_GAIN_AUTO_OFF);
    GXSetFloat(handle_, GX_FLOAT_GAIN, gain_);
    GXSetEnum(handle_, GX_ENUM_BALANCE_WHITE_AUTO, GX_BALANCE_WHITE_AUTO_CONTINUOUS);
    GXSetEnum(handle_, GX_ENUM_BLACKLEVEL_AUTO, GX_BLACKLEVEL_AUTO_OFF);
    GXSetEnum(handle_, GX_ENUM_ACQUISITION_FRAME_RATE_MODE, GX_ACQUISITION_FRAME_RATE_MODE_OFF);

    bool implemented = false;
    GXIsImplemented(handle_, GX_INT_GEV_PACKETSIZE, &implemented);
    if (implemented) GXSetInt(handle_, GX_INT_GEV_PACKETSIZE, 1500);

    GXRegisterCaptureCallback(handle_, nullptr, OnFrameCallback);
    GXSendCommand(handle_, GX_COMMAND_ACQUISITION_START);
    capturing_ = true;
    last_fps_time_ = std::chrono::steady_clock::now();
}

DaHeng::~DaHeng()
{
    if (handle_) {
        if (capturing_) {
            GXSendCommand(handle_, GX_COMMAND_ACQUISITION_STOP);
            GXUnregisterCaptureCallback(handle_);
        }
        GXCloseDevice(handle_);
    }
    GXCloseLib();
    g_daheng_instance = nullptr;
}

void DaHeng::read(cv::Mat& img, std::chrono::steady_clock::time_point& ts)
{
    CameraData data;
    queue_.pop(data);
    img = data.img;
    ts = data.timestamp;
}

} // namespace io