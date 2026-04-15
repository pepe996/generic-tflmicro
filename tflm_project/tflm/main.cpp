#include "tensorflow/lite/core/c/common.h"
#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/micro/recording_micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <numeric>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <gpiod.h>

#define INPUT_SIZE (240 * 320 * 3)
#define OUTPUT_SIZE (4420 * 2)
#ifndef CONSUMER
#define CONSUMER	"Consumer"
#endif
#define WIDTH 0
#define HEIGHT 1

typedef struct{
    cv::Rect box;
    float confidence;
    int index;
} BBOX;

int8_t output_data0[OUTPUT_SIZE * 2];
int8_t output_data1[OUTPUT_SIZE];

cv::Size input_size = cv::Size(320, 240);
cv::Size image_size;
float conf_threshold = 0.6;
float center_variance = 0.1;
float size_variance = 0.2;
int nms_max_output_size = 200;
float nms_iou_threshold = 0.3;
cv::Mat anchors_xy, anchors_wh;

int box_count=0;

namespace{
    using TFLiteOpResolver = tflite::MicroMutableOpResolver<30>;

    TfLiteStatus RegisterOps(TFLiteOpResolver &op_resolver){
        TF_LITE_ENSURE_STATUS(op_resolver.AddReshape());
        TF_LITE_ENSURE_STATUS(op_resolver.AddTranspose());
        TF_LITE_ENSURE_STATUS(op_resolver.AddAdd());
        TF_LITE_ENSURE_STATUS(op_resolver.AddMul());
        TF_LITE_ENSURE_STATUS(op_resolver.AddMean());
        TF_LITE_ENSURE_STATUS(op_resolver.AddConv2D());
        TF_LITE_ENSURE_STATUS(op_resolver.AddExp());
        TF_LITE_ENSURE_STATUS(op_resolver.AddLog());
        TF_LITE_ENSURE_STATUS(op_resolver.AddTanh());
        TF_LITE_ENSURE_STATUS(op_resolver.AddSquaredDifference());
        TF_LITE_ENSURE_STATUS(op_resolver.AddRsqrt());
        TF_LITE_ENSURE_STATUS(op_resolver.AddSub());
        TF_LITE_ENSURE_STATUS(op_resolver.AddFullyConnected());
        TF_LITE_ENSURE_STATUS(op_resolver.AddConcatenation());
        TF_LITE_ENSURE_STATUS(op_resolver.AddQuantize());
        TF_LITE_ENSURE_STATUS(op_resolver.AddPad());
        TF_LITE_ENSURE_STATUS(op_resolver.AddTransposeConv());
        TF_LITE_ENSURE_STATUS(op_resolver.AddStridedSlice());
        TF_LITE_ENSURE_STATUS(op_resolver.AddCast());
        TF_LITE_ENSURE_STATUS(op_resolver.AddDequantize());
        TF_LITE_ENSURE_STATUS(op_resolver.AddCos());
        TF_LITE_ENSURE_STATUS(op_resolver.AddSin());

        TF_LITE_ENSURE_STATUS(op_resolver.AddLogistic());
        TF_LITE_ENSURE_STATUS(op_resolver.AddMaxPool2D());
        TF_LITE_ENSURE_STATUS(op_resolver.AddResizeNearestNeighbor());
        TF_LITE_ENSURE_STATUS(op_resolver.AddDepthwiseConv2D());
        TF_LITE_ENSURE_STATUS(op_resolver.AddSoftmax());
        TF_LITE_ENSURE_STATUS(op_resolver.AddSlice());
        TF_LITE_ENSURE_STATUS(op_resolver.AddAveragePool2D());
        TF_LITE_ENSURE_STATUS(op_resolver.AddBatchMatMul());
        return kTfLiteOk;
    }
} // namespace

float get_iou_value(cv::Rect rect1, cv::Rect rect2){
    int xx1, yy1, xx2, yy2;

    xx1 = std::max(rect1.x, rect2.x);
    yy1 = std::max(rect1.y, rect2.y);
    xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    int insection_w, insection_h;
    insection_w = std::max(0, xx2 - xx1 + 1);
    insection_h = std::max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_w) * insection_h;
    union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
    iou = insection_area / union_area;
    return iou;
}

void nms_boxes(std::vector<cv::Rect> &boxes, std::vector<float> &confidences, float confThreshold, float nmsThreshold, std::vector<int> &indices){
    BBOX bbox;
    std::vector<BBOX> bboxes;
    for (int i = 0; i < boxes.size(); i++){
        bbox.box = boxes[i];
        bbox.confidence = confidences[i];
        bbox.index = i;
        bboxes.push_back(bbox);
    }
    std::sort(bboxes.begin(), bboxes.end(), [](const BBOX &a, const BBOX &b)
              { return a.confidence > b.confidence; });

    int updated_size = bboxes.size();
    for (int i = 0; i < updated_size; i++){
        if (bboxes[i].confidence < confThreshold)
            continue;
        indices.push_back(bboxes[i].index);
        for (int j = i + 1; j < updated_size;){
            float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
            if (iou > nmsThreshold){
                bboxes.erase(bboxes.begin() + j);
                //updated_size = bboxes.size();
                updated_size--;
            }
            else{
                j++;
            }
        }
    }
}

void generate_anchors(){
    cv::Mat input_size_mat = (cv::Mat_<float>(2, 1) << input_size.width, input_size.height);
    // std::cout << input_size_mat << std::endl;
    //  Define the feature maps and min boxes
    std::vector<std::vector<int>> feature_maps = {{40, 30}, {20, 15}, {10, 8}, {5, 4}};
    std::vector<std::vector<int>> min_boxes = {{10, 16, 24}, {32, 48}, {64, 96}, {128, 192, 256}};
    std::vector<cv::Mat> anchors;
    for (int i = 0; i < feature_maps.size(); i++){
        const std::vector<int> &feature_map = feature_maps[i];
        const std::vector<int> &min_box = min_boxes[i];

        // Step 1: Create wh_grid
        cv::Mat min_box_mat = cv::Mat(min_box).t();
        min_box_mat.convertTo(min_box_mat, CV_32F);
        cv::Mat wh_grid;
        cv::divide(cv::repeat(min_box_mat, feature_map.size(), 1), cv::repeat(input_size_mat, 1, min_box.size()), wh_grid);
        cv::Mat wh_grid_rep = cv::repeat(wh_grid.t(), feature_map[WIDTH] * feature_map[HEIGHT], 1);
        // std::cout << wh_grid_rep.size() << std::endl;

        // Step 2: Create xy_grid
        cv::Mat x_grid, y_grid;
        std::vector<int> x_range(feature_map[WIDTH]);
        std::vector<int> y_range(feature_map[HEIGHT]);
        std::iota(x_range.begin(), x_range.end(), 0);
        std::iota(y_range.begin(), y_range.end(), 0);

        cv::repeat(cv::Mat(x_range).t(), feature_map[HEIGHT], 1, x_grid);
        cv::repeat(cv::Mat(y_range), 1, feature_map[WIDTH], y_grid);

        x_grid.convertTo(x_grid, CV_32F);
        y_grid.convertTo(y_grid, CV_32F);

        cv::Mat xy_grid[2] = {x_grid + 0.5, y_grid + 0.5};
        xy_grid[WIDTH] /= feature_map[WIDTH];
        xy_grid[HEIGHT] /= feature_map[HEIGHT];
        // std::cout << xy_grid[0] << std::endl << xy_grid[1] << std::endl;

        cv::Mat stacked_xy_grid;
        cv::merge(xy_grid, 2, stacked_xy_grid);
        stacked_xy_grid = stacked_xy_grid.reshape(1, stacked_xy_grid.total());
        // std::cout << stacked_xy_grid << std::endl;

        cv::Mat tiled_xy_grid;
        cv::repeat(stacked_xy_grid, 1, min_box.size(), tiled_xy_grid);
        // std::cout << tiled_xy_grid.size() << std::endl << tiled_xy_grid << std::endl;
        cv::Mat xy_grid_reshaped = tiled_xy_grid.reshape(1, tiled_xy_grid.total() / 2);

        // Step 3: Concatenate xy_grid and wh_grid
        cv::Mat prior;
        cv::hconcat(xy_grid_reshaped, wh_grid_rep, prior);
        // std::cout << prior.size() << std::endl << prior << std::endl;
        anchors.push_back(prior);
    }
    cv::Mat all_anchors;
    cv::vconcat(anchors, all_anchors);
    // std::cout << all_anchors << std::endl << all_anchors.size() << std::endl;
    cv::Mat xy_anchors = all_anchors.colRange(0, 2);
    cv::Mat wh_anchors = all_anchors.colRange(2, 4);

    cv::Mat clipped_xy_anchors, clipped_wh_anchors;
    cv::min(cv::max(xy_anchors, 0), 1, clipped_xy_anchors);
    cv::min(cv::max(wh_anchors, 0), 1, clipped_wh_anchors);
    anchors_xy = clipped_xy_anchors;
    anchors_wh = clipped_wh_anchors;
    // std::cout << clipped_xy_anchors << std::endl << clipped_wh_anchors << std::endl;
    // std::cout << clipped_xy_anchors.size() << std::endl << clipped_wh_anchors.size() << std::endl;
    // std::cout << anchors_xy.size() << std::endl << anchors_wh.size() << std::endl;
}

cv::Mat decode_regression(const cv::Mat &reg){
    // int rows = OUTPUT_SIZE/2;
    // int cols = 4;
    // cv::Mat reg = cv::Mat(rows, cols, CV_8S, output_data1);
    // reg.convertTo(reg, CV_32F);
    // 1. Bounding box regression: Calculate center_x, center_y
    cv::Mat reg_xy = reg.colRange(0, 2);
    cv::Mat center_xy;
    cv::multiply(reg_xy, center_variance * anchors_wh, center_xy);
    center_xy += anchors_xy;

    // 2. Calculate width and height (center_wh)
    cv::Mat reg_wh = reg.colRange(2, 4);
    cv::Mat exp_reg_wh;
    cv::exp(reg_wh * size_variance, exp_reg_wh);
    cv::Mat center_wh = exp_reg_wh.mul(anchors_wh) / 2.0;

    // 3. Convert from center to corner coordinates (start_xy and end_xy)
    cv::Mat start_xy = center_xy - center_wh;
    cv::Mat end_xy = center_xy + center_wh;

    // 4. Concatenate start_xy and end_xy to form the final boxes
    cv::Mat boxes;
    cv::hconcat(start_xy, end_xy, boxes);

    // 5. Clip the boxes to be within [0, 1]
    cv::min(cv::max(boxes, 0), 1, boxes);

    return boxes;
}

TfLiteStatus LoadQuantModelAndPerformInference(const cv::Mat &input_img){
    // Map the model into a usable data structure. This doesn't involve any
    // copying or parsing, it's a very lightweight operation.
    const tflite::Model *model =
        ::tflite::GetModel(g_model);
    TFLITE_CHECK_EQ(model->version(), TFLITE_SCHEMA_VERSION);

    TFLiteOpResolver op_resolver;
    TF_LITE_ENSURE_STATUS(RegisterOps(op_resolver));

    // Arena size just a round number. The exact arena usage can be determined
    // using the RecordingMicroInterpreter.
    constexpr int kTensorArenaSize = 5000 * 1024;
    uint8_t tensor_arena[kTensorArenaSize];

    tflite::MicroInterpreter interpreter(model, op_resolver, tensor_arena,
                                         kTensorArenaSize);

    TF_LITE_ENSURE_STATUS(interpreter.AllocateTensors());

    // get tensor pointer
    TfLiteTensor *input0 = interpreter.input(0);
    TFLITE_CHECK_NE(input0, nullptr);
    TfLiteTensor *output0 = interpreter.output(0);
    TFLITE_CHECK_NE(output0, nullptr);
    TfLiteTensor *output1 = interpreter.output(1);
    TFLITE_CHECK_NE(output1, nullptr);

    int8_t input_data0[INPUT_SIZE];
    // std::cout << "output0: " << output0->bytes << " bytes  output1: " << output1->bytes << " bytes\n";

    ////////////////////
    // Pre-processing //
    ////////////////////
    // quantize input
    float input0_scale = input0->params.scale;
    int input0_zero_point = input0->params.zero_point;
    // for(int i=0; i<input_img.total()*input_img.channels(); i++){
    //   float pixel = input_img.at<float>(i);
    //   input_data0[i]= static_cast<int8_t>(std::round(pixel / input0_scale) + input0_zero_point);
    // }
    
    for (int i = 0; i < input_img.total(); i++){
        cv::Vec3f pixel = input_img.at<cv::Vec3f>(i / input_img.cols, i % input_img.cols); // Accessing as Vec3f for 3-channel image
        input_data0[i * 3] = static_cast<int8_t>(std::round(pixel[0] / input0_scale) + input0_zero_point);
        input_data0[i * 3 + 1] = static_cast<int8_t>(std::round(pixel[1] / input0_scale) + input0_zero_point);
        input_data0[i * 3 + 2] = static_cast<int8_t>(std::round(pixel[2] / input0_scale) + input0_zero_point);
    }
    // test quantized input data
    for (int i = 0; i < 10; i++){
        printf("input_data0[%d] = %d\n", i, input_data0[i]);
    }
    MicroPrintf("Convert to int8 rgb image!\n");
    // copy to input tensor
    int8_t *inTensorData0 = tflite::GetTensorData<int8_t>(input0);
    memcpy(inTensorData0, input_data0, input0->bytes);

    /////////////////////
    // Model Inference //
    /////////////////////
    // run one invoke & time measurement
    MicroPrintf("Running model inference...");
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    TF_LITE_ENSURE_STATUS(interpreter.Invoke());
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Execution time of one invoke: %f seconds\n\n", cpu_time_used);

    // copy output
    memcpy(&output_data0[0], tflite::GetTensorData<int8_t>(output0), output0->bytes);
    memcpy(&output_data1[0], tflite::GetTensorData<int8_t>(output1), output1->bytes);

    /////////////////////
    // Post-processing //
    /////////////////////
    float output_score_scale = output1->params.scale;
    int output_score_zero_point = output1->params.zero_point;
    float output_box_scale = output0->params.scale;
    int output_box_zero_point = output0->params.zero_point;

    // test dequantized output data
    for (int i = 0; i < 5; i++){
        printf("output_score_data[%d] = %f\n", 2 * i + 1, (float)(output_data1[2 * i + 1] - output_score_zero_point) * output_score_scale);
        printf("output_box_data[%d] = %f\n", 4 * i, (float)(output_data0[4 * i] - output_box_zero_point) * output_box_scale);
        printf("output_box_data[%d] = %f\n", 4 * i + 1, (float)(output_data0[4 * i + 1] - output_box_zero_point) * output_box_scale);
        printf("output_box_data[%d] = %f\n", 4 * i + 2, (float)(output_data0[4 * i + 2] - output_box_zero_point) * output_box_scale);
        printf("output_box_data[%d] = %f\n", 4 * i + 3, (float)(output_data0[4 * i + 3] - output_box_zero_point) * output_box_scale);
    }

    // dequantize output
    cv::Mat output_score_mat(OUTPUT_SIZE / 2, 2, CV_32F);
    cv::Mat output_box_mat(OUTPUT_SIZE / 2, 4, CV_32F);
    for (int i = 0; i < OUTPUT_SIZE / 2; i++){
        output_score_mat.at<float>(i, 0) = (float)(output_data1[2 * i] - output_score_zero_point) * output_score_scale;
        output_score_mat.at<float>(i, 1) = (float)(output_data1[2 * i + 1] - output_score_zero_point) * output_score_scale;
        output_box_mat.at<float>(i, 0) = (float)(output_data0[4 * i] - output_box_zero_point) * output_box_scale;
        output_box_mat.at<float>(i, 1) = (float)(output_data0[4 * i + 1] - output_box_zero_point) * output_box_scale;
        output_box_mat.at<float>(i, 2) = (float)(output_data0[4 * i + 2] - output_box_zero_point) * output_box_scale;
        output_box_mat.at<float>(i, 3) = (float)(output_data0[4 * i + 3] - output_box_zero_point) * output_box_scale;
    }
    MicroPrintf("Output Dequantized!\n");

    cv::Mat boxes = decode_regression(output_box_mat);
    // std::cout << boxes << std::endl << boxes.size() << std::endl;
    cv::Mat scores = output_score_mat.col(1).clone();

    // Create a mask where scores > conf_threshold
    cv::Mat conf_mask;
    cv::compare(scores, conf_threshold, conf_mask, cv::CMP_GT); // Compare each score with conf_threshold
    // Use the mask to filter boxes and scores
    // First, find the non-zero indices in the mask (where the condition is true)
    std::vector<int> indices;
    for (int i = 0; i < conf_mask.rows; i++){
        if (conf_mask.at<uchar>(i))
        {
            indices.push_back(i);
        }
    }
    cv::Mat filtered_boxes, filtered_scores;
    for (int i : indices){
        filtered_boxes.push_back(boxes.row(i));   // Append valid box rows
        filtered_scores.push_back(scores.row(i)); // Append valid score rows
    }
    // Convert filtered_boxes (cv::Mat) to vector<cv::Rect>
    std::vector<cv::Rect> rect_boxes;
    for (int i = 0; i < filtered_boxes.rows; i++){
        int x = (filtered_boxes.at<float>(i, 0)) * image_size.width;
        int y = (filtered_boxes.at<float>(i, 1)) * image_size.height;
        int width = (filtered_boxes.at<float>(i, 2) - filtered_boxes.at<float>(i, 0)) * image_size.width;
        int height = (filtered_boxes.at<float>(i, 3) - filtered_boxes.at<float>(i, 1)) * image_size.height;
        // printf("(x, y, w, h)=(%d, %d, %d, %d)\n",x,y,width,height);
        rect_boxes.emplace_back(cv::Rect(x, y, width, height));
    }

    // Convert filtered_scores (cv::Mat) to vector<float>
    std::vector<float> score_list;
    for (int i = 0; i < filtered_scores.rows; i++){
        score_list.push_back(filtered_scores.at<float>(i, 0));
    }
    std::vector<int> nms_indices;
    // Perform NMS
    nms_boxes(rect_boxes, score_list, conf_threshold, nms_iou_threshold, nms_indices);
    std::vector<cv::Rect> nms_filtered_boxes;
    for (int i : nms_indices){
        nms_filtered_boxes.push_back(rect_boxes[i]);
    }
    MicroPrintf("Successfully filter boxes and scores!\n");

    // Draw bounding box
    cv::Mat out_img = cv::imread("output_frame.png", cv::IMREAD_COLOR);

    for (const auto &box : nms_filtered_boxes){
        box_count++;
        cv::rectangle(out_img, box, cv::Scalar(0, 255, 0), 2); // Draw selected boxes
        printf("Drawing cv::Rect: ");
        std::cout <<"x,y="<< box.tl() << " w,h=" << box.size() << std::endl;
    }
    if(box_count==0){
        MicroPrintf("No faces found!\n");
    }
    else if(box_count==1){
        MicroPrintf("Successfully output 1 bounding box!\n");
    }
    else{
        //MicroPrintf("Successfully output bounding boxes!\n");
        std::cout << "Successfully output " << box_count << " bounding boxes!\n\n";
    }
    cv::imwrite("output_frame_bbox.png", out_img);

    printf("arena_used_bytes = %ld\n", interpreter.arena_used_bytes());
    return kTfLiteOk;
}

int main(int argc, char *argv[]){
    
    cv::Mat frame;
    //read a frame from camera or an image file
    if (argc < 2){
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cout << "\nError: Cannot open camera\n";
            return 1;
        }
        else{
        std::cout << "\nSuccessfully open camera 0\n" ;
        }
        bool ret = cap.read(frame); // or cap >> frame;
        if (!ret) {
            std::cout << "\nError: Can't receive frame (stream end?). Exiting ...\n";
            return 1;
        }
        else{
        std::cout << "\nSuccessfully receive one frame from camera!\n";
        }
    }
    else{
        frame = cv::imread(argv[1], cv::IMREAD_COLOR);
        // Check if the image was loaded successfully
        if (frame.empty()){
            std::cerr << "\nError: Could not open or find the image from " << argv[1] << std::endl;
            return -1;
        }
        else{
            std::cout << "\nSuccessfully load image from " << argv[1] << std::endl;
        }
    }
    image_size = frame.size();
    cv::imwrite("output_frame.png", frame);


    ////////////////////
    // Pre-processing //
    ////////////////////
    cv::Mat resized_img, image_rgb;
    cv::resize(frame, resized_img, cv::Size(320, 240));
    cv::cvtColor(resized_img, image_rgb, cv::COLOR_BGR2RGB);
    image_rgb.convertTo(image_rgb, CV_32FC3, 1.0 / 255.0);
    cv::Mat image_norm;
    cv::normalize(image_rgb, image_norm, -1.0, 1.0, cv::NORM_MINMAX);
    MicroPrintf("\nSuccessfully normalize rgb image!\n");
    generate_anchors();


    // tflite::InitializeTarget();
    ///////////////////////////////////////
    // Model Inference & Post-processing //
    ///////////////////////////////////////
    TF_LITE_ENSURE_STATUS(LoadQuantModelAndPerformInference(image_norm));


    //////////////////
    // Ouput Signal //
    //////////////////
    const char *chipname = "gpiochip0";
	unsigned int line_num = 120;	// GPIO Pin P15_0
	unsigned int val;
	struct gpiod_chip *chip;
	struct gpiod_line *line;
	int i, ret;

	chip = gpiod_chip_open_by_name(chipname);
	if (!chip) {
		perror("Open chip failed\n");
		return -1;
	}

	line = gpiod_chip_get_line(chip, line_num);
	if (!line) {
		perror("Get line failed\n");
		gpiod_chip_close(chip);
		return -1;
	}

	ret = gpiod_line_request_output(line, CONSUMER, 0);
	if (ret < 0) {
		perror("Request line as output failed\n");
		gpiod_line_release(line);
	}
    if(box_count==0){
        val = 0;
    }
    else{
        val = 1;
    }
    ret = gpiod_line_set_value(line, val);
    if (ret < 0) {
        perror("Set line output failed\n");
        gpiod_line_release(line);
    }
    printf("Output %u on line #%u (GPIO Pin P15_0)\n", val, line_num);
    sleep(1);


    MicroPrintf("~~~ALL TESTS PASSED~~~\n");
    return kTfLiteOk;
}
