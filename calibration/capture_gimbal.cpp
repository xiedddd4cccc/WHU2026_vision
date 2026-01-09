#include <fmt/core.h>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/demo.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/calib_images   | 输出文件夹路径   }";

// 棋盘格参数（必须与 capture_loop 中一致）
constexpr int BOARD_COLS = 6; // 物理列数
constexpr int BOARD_ROWS = 8; // 物理行数
constexpr float SQUARE_SIZE_M = 0.029f; // 29mm

int capture_loop(const std::string & config_path, const std::string & output_folder)
{
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  const cv::Size patternSize(BOARD_COLS - 1, BOARD_ROWS - 1); // 内部角点数
  const float squareSize = SQUARE_SIZE_M;
  const cv::Scalar borderColor(0, 255, 0);
  const int borderThickness = 2;

  int count = 0;
  while (true) {
    camera.read(img, timestamp);
    
    auto display_img = img.clone();
    tools::draw_text(display_img, fmt::format("Count: {}", count), {40, 40}, {0, 255, 255});
    tools::draw_text(display_img, "Press 's' to save, 'q' to quit", {40, 80}, {0, 255, 255});
    tools::draw_text(display_img,
                     fmt::format("Board: {}x{} {}mm", BOARD_COLS, BOARD_ROWS,
                                 static_cast<int>(squareSize * 1000)),
                     {40, 120}, {0, 255, 255});

    std::vector<cv::Point2f> corners;
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    bool success = cv::findChessboardCorners(
        gray, patternSize, corners,
        cv::CALIB_CB_ADAPTIVE_THRESH |
        cv::CALIB_CB_NORMALIZE_IMAGE |
        cv::CALIB_CB_FAST_CHECK);

    if (success) {
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);
        cv::drawChessboardCorners(display_img, patternSize, corners, success);

        cv::Point2f topLeft = corners[0];
        cv::Point2f topRight = corners[patternSize.width - 1];
        cv::Point2f bottomRight = corners.back();
        cv::Point2f bottomLeft = corners[corners.size() - patternSize.width];

        std::vector<cv::Point> borderPoints = {
            static_cast<cv::Point>(topLeft),
            static_cast<cv::Point>(topRight),
            static_cast<cv::Point>(bottomRight),
            static_cast<cv::Point>(bottomLeft)
        };
        cv::polylines(display_img, borderPoints, true, borderColor, borderThickness);

        std::string sizeInfo = fmt::format("{}x{} {}mm", BOARD_COLS, BOARD_ROWS,
                                           static_cast<int>(squareSize * 1000));
        cv::putText(display_img, sizeInfo,
                    cv::Point(static_cast<int>(topLeft.x), static_cast<int>(topLeft.y - 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, borderColor, 1);

        tools::draw_text(display_img, "Detection: SUCCESS", {40, 160}, {0, 255, 0});
    } else {
        tools::draw_text(display_img, "Detection: FAILED", {40, 160}, {0, 0, 255});
    }

    cv::resize(display_img, display_img, {}, 0.5, 0.5);
    cv::imshow("Press s to save, q to quit", display_img);
    int key = cv::waitKey(1);

    if (key == 'q' || key == 27) {
        break;
    }

    if (key == 's') {
        if (success) {
            count++;
            auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
            cv::imwrite(img_path, img);
            tools::logger()->info("[{}] Saved image: {}", count, img_path);
            
            // 短暂显示 SAVED 提示
            cv::Mat notice = display_img.clone();
            cv::putText(notice, "SAVED!", 
                        cv::Point(notice.cols/2 - 50, notice.rows/2),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Press s to save, q to quit", notice);
            cv::waitKey(300);
        } else {
            tools::logger()->warn("Cannot save: chessboard not detected!");
        }
    }
  }

  cv::destroyAllWindows();
  return count;
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");

  std::filesystem::create_directories(output_folder);

  // 日志中使用常量，避免作用域错误
  tools::logger()->info("棋盘格尺寸: {} 列 × {} 行 (内部角点 {}×{}), {}mm 方格",
                        BOARD_COLS, BOARD_ROWS,
                        BOARD_COLS - 1, BOARD_ROWS - 1,
                        static_cast<int>(SQUARE_SIZE_M * 1000));
  tools::logger()->info("输出文件夹: {}", output_folder);
  tools::logger()->info("按 's' 保存（需检测成功），按 'q' 或 ESC 退出");

  int saved_count = capture_loop(config_path, output_folder);

  tools::logger()->info("程序结束，共保存 {} 张图像", saved_count);

  return 0;
}