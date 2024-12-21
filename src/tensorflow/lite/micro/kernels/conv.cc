/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// #include "tensorflow/lite/micro/kernels/conv.h"

// #include "tensorflow/lite/c/builtin_op_data.h"
// #include "tensorflow/lite/c/common.h"
// #include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
// #include "tensorflow/lite/kernels/internal/reference/conv.h"
// #include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
// #include "tensorflow/lite/kernels/kernel_util.h"
// #include "tensorflow/lite/micro/kernels/kernel_util.h"
// #include "tensorflow/lite/micro/micro_log.h"

// namespace tflite {
// namespace {

// TfLiteStatus ConvEval(TfLiteContext* context, TfLiteNode* node) {
//   const TfLiteEvalTensor* input =
//       tflite::micro::GetEvalInput(context, node, kConvInputTensor);
//   const TfLiteEvalTensor* filter =
//       tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
//   const TfLiteEvalTensor* bias =
//       (NumInputs(node) == 3)
//           ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
//           : nullptr;
//   TfLiteEvalTensor* output =
//       tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

//   TFLITE_DCHECK(node->builtin_data != nullptr);
//   const auto& params =
//       *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
//   TFLITE_DCHECK(node->user_data != nullptr);
//   const auto& data = *(static_cast<const OpDataConv*>(node->user_data));

//   switch (input->type) {  // Already know in/out types are same.
//     case kTfLiteFloat32: {
//       tflite::reference_ops::Conv(
//           ConvParamsFloat(params, data), tflite::micro::GetTensorShape(input),
//           tflite::micro::GetTensorData<float>(input),
//           tflite::micro::GetTensorShape(filter),
//           tflite::micro::GetTensorData<float>(filter),
//           tflite::micro::GetTensorShape(bias),
//           tflite::micro::GetOptionalTensorData<float>(bias),
//           tflite::micro::GetTensorShape(output),
//           tflite::micro::GetTensorData<float>(output),
//           tflite::micro::GetTensorShape(nullptr), nullptr);
//       break;
//     }
//     case kTfLiteInt16: {
//       if (bias == nullptr || bias->type == kTfLiteInt32) {
//         reference_integer_ops::ConvPerChannel(
//             ConvParamsQuantized(params, data),
//             data.per_channel_output_multiplier, data.per_channel_output_shift,
//             tflite::micro::GetTensorShape(input),
//             tflite::micro::GetTensorData<int16_t>(input),
//             tflite::micro::GetTensorShape(filter),
//             tflite::micro::GetTensorData<int8_t>(filter),
//             tflite::micro::GetTensorShape(bias),
//             tflite::micro::GetOptionalTensorData<std::int32_t>(bias),
//             tflite::micro::GetTensorShape(output),
//             tflite::micro::GetTensorData<int16_t>(output));
//       } else if (bias->type == kTfLiteInt64) {
//         reference_integer_ops::ConvPerChannel(
//             ConvParamsQuantized(params, data),
//             data.per_channel_output_multiplier, data.per_channel_output_shift,
//             tflite::micro::GetTensorShape(input),
//             tflite::micro::GetTensorData<int16_t>(input),
//             tflite::micro::GetTensorShape(filter),
//             tflite::micro::GetTensorData<int8_t>(filter),
//             tflite::micro::GetTensorShape(bias),
//             tflite::micro::GetOptionalTensorData<std::int64_t>(bias),
//             tflite::micro::GetTensorShape(output),
//             tflite::micro::GetTensorData<int16_t>(output));
//       } else {
//         MicroPrintf("Bias type %s (%d) not supported.",
//                     TfLiteTypeGetName(bias->type), bias->type);
//         return kTfLiteError;
//       }
//       break;
//     }
//     case kTfLiteInt8: {
//       switch (filter->type) {
//         case kTfLiteInt4: {
//           int8_t* unpacked_filter_data = static_cast<int8_t*>(
//               context->GetScratchBuffer(context, data.filter_buffer_index));
//           tflite::tensor_utils::UnpackDenseInt4IntoInt8(
//               tflite::micro::GetTensorData<int8_t>(filter),
//               tflite::micro::GetTensorShape(filter).FlatSize(),
//               unpacked_filter_data);
//           reference_integer_ops::ConvPerChannel(
//               ConvParamsQuantized(params, data),
//               data.per_channel_output_multiplier, data.per_channel_output_shift,
//               tflite::micro::GetTensorShape(input),
//               tflite::micro::GetTensorData<int8_t>(input),
//               tflite::micro::GetTensorShape(filter), unpacked_filter_data,
//               tflite::micro::GetTensorShape(bias),
//               tflite::micro::GetOptionalTensorData<int32_t>(bias),
//               tflite::micro::GetTensorShape(output),
//               tflite::micro::GetTensorData<int8_t>(output));
//           break;
//         }
//         case kTfLiteInt8: {
//           reference_integer_ops::ConvPerChannel(
//               ConvParamsQuantized(params, data),
//               data.per_channel_output_multiplier, data.per_channel_output_shift,
//               tflite::micro::GetTensorShape(input),
//               tflite::micro::GetTensorData<int8_t>(input),
//               tflite::micro::GetTensorShape(filter),
//               tflite::micro::GetTensorData<int8_t>(filter),
//               tflite::micro::GetTensorShape(bias),
//               tflite::micro::GetOptionalTensorData<int32_t>(bias),
//               tflite::micro::GetTensorShape(output),
//               tflite::micro::GetTensorData<int8_t>(output));
//           break;
//         }
//         default:
//           MicroPrintf("Weight type %s (%d) not supported.",
//                       TfLiteTypeGetName(filter->type), filter->type);
//           return kTfLiteError;
//       }
//       break;
//     }
//     default:
//       MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
//                   input->type);
//       return kTfLiteError;
//   }
//   return kTfLiteOk;
// }

// }  // namespace

// TFLMRegistration Register_CONV_2D() {
//   return tflite::micro::RegisterOp(ConvInit, ConvPrepare, ConvEval);
// }

// }  // namespace tflite
#include "tensorflow/lite/micro/kernels/conv.h"

#include "muriscv_nn_types.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#define USE_PEXT 1
#ifdef USE_PEXT
#include <nds_intrinsic.h>
#define PACK_Q7x4_32x1(v0, v1, v2, v3) ((__rv__pkbb16((uint8_t)(v0), (uint8_t)(v2))) | ((__rv__pkbb16((uint8_t)(v1), (uint8_t)(v3))) << 8))
#endif

int32_t muriscv_nn_requantize(const int32_t val, const int32_t multiplier, const int32_t shift)
{
    const int64_t total_shift = 31 - shift;
    const int64_t new_val = val * (int64_t)multiplier;

    int32_t result = new_val >> (total_shift - 1);
    result = (result + 1) >> 1;
    return result;
}

q31_t muriscv_nn_read_q15x2_ia_fast(const q15_t **in_q15)
{
    q31_t val;
    val = (*((uint32_t*)(*in_q15)));
    *in_q15 += 2;

    return val;
}

q31_t muriscv_nn_read_q7x4_ia_fast(const q7_t **in_q7)
{
    q31_t val;
    val = (*((q31_t*)(*in_q7)));
    *in_q7 += 4;
    return val;
}

q31_t muriscv_nn_read_q15x2_ia_aligned(const q15_t **in_q15, const uint8_t alignment)
{
    q31_t val;
    if(alignment == 0)
    {
         val = (*((uint32_t*)(*in_q15)));       
    }
    else
    {
         val = (uint32_t)((*((uint64_t*)(*in_q15 - 1))) >> 16); 
    }
    *in_q15 += 2;

    return val;
}

q31_t muriscv_nn_read_q7x4_ia_aligned(const q7_t **in_q7, const uint8_t alignment, const uint8_t alignment_bits)
{
    q31_t val;
    if (alignment == 0)
    {
        val = (*((q31_t*)(*in_q7)));
    }
    else
    {
        val = (uint32_t)((*((uint64_t*)(*in_q7 - alignment))) >> (alignment_bits));
    }
    *in_q7 += 4;
    return val;
}

void muriscv_nn_write_q7x4_ia(q7_t **in, q31_t value)
{
    memcpy(*in, &value, 4);
    *in += 4;
}

void muriscv_nn_write_q15x2_ia(int16_t **dest_q15, int32_t src_q31)
{
    int32_t val = src_q31;

    memcpy(*dest_q15, &val, 4);
    *dest_q15 += 2;
}


int32_t muriscv_nn_convolve_s8_get_buffer_size(const muriscv_nn_dims *input_dims, const muriscv_nn_dims *filter_dims)
{

  const int32_t rhs_cols = filter_dims->w * filter_dims->h * input_dims->c;
  const int32_t remainder = rhs_cols % 4;
  const int32_t aligned_rhs_cols = remainder != 0 ? rhs_cols + 4 - remainder : rhs_cols;
  return (2 * aligned_rhs_cols) * (int32_t)sizeof(int16_t);

}
void muriscv_nn_q7_to_q15_with_offset(const int8_t *src, int16_t *dst, int32_t block_size, int16_t offset)
{
  // int32_t in_s8x4;
  // int32_t in_s16x2_1;
  // int32_t in_s16x2_2;
  // int32_t block_cnt = block_size >> 2;

  // /* Compute 4 outputs at a time. */
  // const int32_t offset_s16x2 = __rv__pkbb16(offset, offset);
  // while (block_cnt > 0)
  // {
  //     in_s8x4 = muriscv_nn_read_q7x4_ia_fast(&src);

  //     in_s16x2_1 = __rv__kadd16(offset_s16x2, __rv__sunpkd810(in_s8x4));
  //     in_s16x2_2 = __rv__kadd16(offset_s16x2, __rv__sunpkd832(in_s8x4));

  //     muriscv_nn_write_q15x2_ia(&dst, in_s16x2_1);
  //     muriscv_nn_write_q15x2_ia(&dst, in_s16x2_2);

  //     block_cnt--;
  // }

  // /* Handle left over samples. */
  // block_cnt = block_size % 4;
  
  int32_t block_cnt = block_size;
  // Serial.print(block_cnt);
  // Serial.print(" ");
  while (block_cnt > 0)
  {
    *dst++ = (int16_t)*src++ + offset;

    /* Decrement the loop counter */
    block_cnt--;
  }
}

q7_t *muriscv_nn_mat_mult_kernel_s8_s16(const q7_t *input_a,
                                        const q15_t *input_b,
                                        const uint16_t output_ch,
                                        const int32_t *out_shift,
                                        const int32_t *out_mult,
                                        const int32_t out_offset,
                                        const int16_t activation_min,
                                        const int16_t activation_max,
                                        const uint16_t num_col_a,
                                        const int32_t *const output_bias,
                                        q7_t *out_0)
{
#if !defined(USE_VEXT)
    /* set up the second output pointers */
    q7_t *out_1 = out_0 + output_ch;
    const int32_t *bias = output_bias;
    #if defined(USE_PEXT)
    const q7_t b1_align = num_col_a % 2;
    uint8_t a1_align = (num_col_a % 4);
    uint8_t a1_align_bits = a1_align << 3;
    uint8_t a2_align = (num_col_a % 2) << 1;
    uint8_t a2_align_bits = a2_align << 3;
    uint8_t a3_align = (4 - (num_col_a % 4)) % 4;
    uint8_t a3_align_bits = a3_align << 3;

    // uint32_t out_offset_s8x4 = PACK_Q7x4_32x1(out_offset, out_offset, out_offset, out_offset);
    // uint32_t activation_min_s8x4 = PACK_Q7x4_32x1(activation_min, activation_min, activation_min, activation_min);
    // uint32_t activation_max_s8x4 = PACK_Q7x4_32x1(activation_max, activation_max, activation_max, activation_max);
    #endif
    
    uint16_t row_count = output_ch >> 2;
    
    const q7_t *ip_a0 = input_a;
    /* this loop over rows in A */
    while (row_count)
    {
        /* setup pointers for B */
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;

        
        /* align the pointers for A */
        const q7_t *ip_a1 = ip_a0 + num_col_a;
        const q7_t *ip_a2 = ip_a1 + num_col_a;
        const q7_t *ip_a3 = ip_a2 + num_col_a;


        q31_t ch_0_out_0 = 0;
        q31_t ch_0_out_1 = 0;



        q31_t ch_1_out_0 = 0;
        q31_t ch_1_out_1 = 0;

        q31_t ch_2_out_0 = 0;
        q31_t ch_2_out_1 = 0;

        q31_t ch_3_out_0 = 0;
        q31_t ch_3_out_1 = 0;
        /* Init accumulator with bias for channel N and N + 1 */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;

            ch_1_out_0 = *bias;
            ch_1_out_1 = *bias++;
            //new

            ch_2_out_0 = *bias;
            ch_2_out_1 = *bias++;

            ch_3_out_0 = *bias;
            ch_3_out_1 = *bias++;
        }

#if defined(USE_PEXT)
        uint16_t col_count = num_col_a / 4;

        /* accumulate over the vector */
        while (col_count)
        {
            //Access always word aligned
            q31_t b01 = muriscv_nn_read_q15x2_ia_fast(&ip_b0);
            
            //Access is word aligned when num_col_a is even              
            q31_t b11 = muriscv_nn_read_q15x2_ia_aligned(&ip_b1, b1_align);

            
            
            //access word aligned when location is multiple of 4
            q31_t inA = muriscv_nn_read_q7x4_ia_fast(&ip_a0);
    
            q31_t a01 = __rv__sunpkd810(inA);
            q31_t a02 = __rv__sunpkd832(inA);

            ch_0_out_0 = __rv__kmada(ch_0_out_0, a01, b01);
            ch_0_out_1 = __rv__kmada(ch_0_out_1, a01, b11);
            
            //access word aligned when location is multiple of 4
            inA = muriscv_nn_read_q7x4_ia_aligned(&ip_a1, a1_align, a1_align_bits);
            
            q31_t a11 = __rv__sunpkd810(inA);
            q31_t a12 = __rv__sunpkd832(inA);

            ch_1_out_0 = __rv__kmada(ch_1_out_0, a11, b01);
            ch_1_out_1 = __rv__kmada(ch_1_out_1, a11, b11);

            //access word aligned when location is multiple of 4
            inA = muriscv_nn_read_q7x4_ia_aligned(&ip_a2, a2_align, a2_align_bits);
            
            q31_t a21 = __rv__sunpkd810(inA);
            q31_t a22 = __rv__sunpkd832(inA);

            ch_2_out_0 = __rv__kmada(ch_2_out_0, a21, b01);
            ch_2_out_1 = __rv__kmada(ch_2_out_1, a21, b11);

            //access word aligned when location is multiple of 4
            inA = muriscv_nn_read_q7x4_ia_aligned(&ip_a3, a3_align, a3_align_bits);
            
            q31_t a31 = __rv__sunpkd810(inA);
            q31_t a32 = __rv__sunpkd832(inA);

            ch_3_out_0 = __rv__kmada(ch_3_out_0, a31, b01);
            ch_3_out_1 = __rv__kmada(ch_3_out_1, a31, b11);


            //access always word aligned
            q31_t b02 = muriscv_nn_read_q15x2_ia_fast(&ip_b0);
            //Access is word aligned when num_col_a is even         
            q31_t b12 = muriscv_nn_read_q15x2_ia_aligned(&ip_b1, b1_align);

            ch_0_out_0 = __rv__kmada(ch_0_out_0, a02, b02);
            ch_0_out_1 = __rv__kmada(ch_0_out_1, a02, b12);

            ch_1_out_0 = __rv__kmada(ch_1_out_0, a12, b02);
            ch_1_out_1 = __rv__kmada(ch_1_out_1, a12, b12);

            ch_2_out_0 = __rv__kmada(ch_2_out_0, a22, b02);
            ch_2_out_1 = __rv__kmada(ch_2_out_1, a22, b12);

            ch_3_out_0 = __rv__kmada(ch_3_out_0, a32, b02);
            ch_3_out_1 = __rv__kmada(ch_3_out_1, a32, b12);

            
            
            

            col_count--;
        } /* while over col_count */
        col_count = num_col_a & 0x3;
#else
        uint16_t col_count = num_col_a;
#endif
        while (col_count)
        {
            
            q7_t a0 = *ip_a0++;
            q15_t b0 = *ip_b0++;
            q7_t a1 = *ip_a1++;
            q15_t b1 = *ip_b1++;

            //new
            q7_t a2 = *ip_a2++;
            q7_t a3 = *ip_a3++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            ch_1_out_0 += a1 * b0;
            ch_1_out_1 += a1 * b1;

            //new
            ch_2_out_0 += a2 * b0;
            ch_2_out_1 += a2 * b1;
            ch_3_out_0 += a3 * b0;
            ch_3_out_1 += a3 * b1;

            
            col_count--;
            
        } /* while over col_count */
        

        #if defined(USE_PEXT)          //SAME ISSUE AS OTHER USES OF ADD8 for offset  TODO:  Change to Add16
                                       //Investigate why this didnt occur before?  ADD8 Saturation?
        ch_0_out_0 = muriscv_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_1 = muriscv_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        out_mult++;
        out_shift++;

        ch_1_out_0 = muriscv_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_1 = muriscv_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        out_mult++;
        out_shift++;

        ch_2_out_0 = muriscv_nn_requantize(ch_2_out_0, *out_mult, *out_shift);
        ch_2_out_0 += out_offset;
        ch_2_out_1 = muriscv_nn_requantize(ch_2_out_1, *out_mult, *out_shift);
        ch_2_out_1 += out_offset;
        out_mult++;
        out_shift++;

        ch_3_out_0 = muriscv_nn_requantize(ch_3_out_0, *out_mult, *out_shift);
        ch_3_out_0 += out_offset;
        ch_3_out_1 = muriscv_nn_requantize(ch_3_out_1, *out_mult, *out_shift);
        ch_3_out_1 += out_offset;
        out_mult++;
        out_shift++;
        
        
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        
        
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        
        
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        
        
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        
        ch_2_out_0 = MAX(ch_2_out_0, activation_min);
        ch_2_out_0 = MIN(ch_2_out_0, activation_max);
        
        ch_2_out_1 = MAX(ch_2_out_1, activation_min);
        ch_2_out_1 = MIN(ch_2_out_1, activation_max);
        
        ch_3_out_0 = MAX(ch_3_out_0, activation_min);
        ch_3_out_0 = MIN(ch_3_out_0, activation_max);
        
        ch_3_out_1 = MAX(ch_3_out_1, activation_min);
        ch_3_out_1 = MIN(ch_3_out_1, activation_max);



        uint32_t packed_out = PACK_Q7x4_32x1(ch_0_out_0, ch_1_out_0, ch_2_out_0, ch_3_out_0);
        //packed_out = __rv__add8(packed_out, out_offset_s8x4);
        //packed_out = __rv__smax8(packed_out, activation_min_s8x4);
        //packed_out = __rv__smin8(packed_out, activation_max_s8x4);
        muriscv_nn_write_q7x4_ia(&out_0, packed_out);

        packed_out = PACK_Q7x4_32x1(ch_0_out_1, ch_1_out_1, ch_2_out_1, ch_3_out_1);
        //packed_out = __rv__add8(packed_out, out_offset_s8x4);
        //packed_out = __rv__smax8(packed_out, activation_min_s8x4);
        //packed_out = __rv__smin8(packed_out, activation_max_s8x4);
        muriscv_nn_write_q7x4_ia(&out_1, packed_out);


        #else
        ch_0_out_0 = muriscv_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (q7_t)ch_0_out_0;

        ch_0_out_1 = muriscv_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (q7_t)ch_0_out_1;
        out_mult++;
        out_shift++;

        ch_1_out_0 = muriscv_nn_requantize(ch_1_out_0, *out_mult, *out_shift);
        ch_1_out_0 += out_offset;
        ch_1_out_0 = MAX(ch_1_out_0, activation_min);
        ch_1_out_0 = MIN(ch_1_out_0, activation_max);
        *out_0++ = (q7_t)ch_1_out_0;

        ch_1_out_1 = muriscv_nn_requantize(ch_1_out_1, *out_mult, *out_shift);
        ch_1_out_1 += out_offset;
        ch_1_out_1 = MAX(ch_1_out_1, activation_min);
        ch_1_out_1 = MIN(ch_1_out_1, activation_max);
        *out_1++ = (q7_t)ch_1_out_1;
        out_mult++;
        out_shift++;

        ch_2_out_0 = muriscv_nn_requantize(ch_2_out_0, *out_mult, *out_shift);
        ch_2_out_0 += out_offset;
        ch_2_out_0 = MAX(ch_2_out_0, activation_min);
        ch_2_out_0 = MIN(ch_2_out_0, activation_max);
        *out_0++ = (q7_t)ch_2_out_0;

        ch_2_out_1 = muriscv_nn_requantize(ch_2_out_1, *out_mult, *out_shift);
        ch_2_out_1 += out_offset;
        ch_2_out_1 = MAX(ch_2_out_1, activation_min);
        ch_2_out_1 = MIN(ch_2_out_1, activation_max);
        *out_1++ = (q7_t)ch_2_out_1;
        out_mult++;
        out_shift++;
        
        ch_3_out_0 = muriscv_nn_requantize(ch_3_out_0, *out_mult, *out_shift);
        ch_3_out_0 += out_offset;
        ch_3_out_0 = MAX(ch_3_out_0, activation_min);
        ch_3_out_0 = MIN(ch_3_out_0, activation_max);
        *out_0++ = (q7_t)ch_3_out_0;

        ch_3_out_1 = muriscv_nn_requantize(ch_3_out_1, *out_mult, *out_shift);
        ch_3_out_1 += out_offset;
        ch_3_out_1 = MAX(ch_3_out_1, activation_min);
        ch_3_out_1 = MIN(ch_3_out_1, activation_max);
        *out_1++ = (q7_t)ch_3_out_1;
        out_mult++;
        out_shift++;
        #endif

        /* skip rows */
        ip_a0 += (num_col_a * 3);
        row_count--;
    }

    /* compute the last three rows if any */
    uint8_t remaining = output_ch % 4;
    while (remaining)
    {
        /* setup pointers for B */
        const q15_t *ip_b0 = input_b;
        const q15_t *ip_b1 = ip_b0 + num_col_a;

        q31_t ch_0_out_0 = 0;
        q31_t ch_0_out_1 = 0;

        /* load the bias */
        if (bias)
        {
            ch_0_out_0 = *bias;
            ch_0_out_1 = *bias++;
        }

#if defined(USE_PEXT)
        uint16_t col_count = num_col_a >> 2;
        while (col_count)
        {
            //possible optimization here
            q31_t b0 = muriscv_nn_read_q15x2_ia_fast(&ip_b0);
            q31_t b1 = muriscv_nn_read_q15x2_ia_fast(&ip_b1);

            q31_t inA = muriscv_nn_read_q7x4_ia_fast(&ip_a0);
            q31_t a01 = __rv__sunpkd810(inA);
            q31_t a02 = __rv__sunpkd832(inA);

            ch_0_out_0 = __rv__kmada(ch_0_out_0, a01, b0);
            ch_0_out_1 = __rv__kmada(ch_0_out_1, a01, b1);

            b0 = muriscv_nn_read_q15x2_ia_fast(&ip_b0);
            b1 = muriscv_nn_read_q15x2_ia_fast(&ip_b1);
            ch_0_out_0 = __rv__kmada(ch_0_out_0, a02, b0);
            ch_0_out_1 = __rv__kmada(ch_0_out_1, a02, b1);

            col_count--;
        }
        col_count = num_col_a & 0x3;
#else
        uint16_t col_count = num_col_a;
#endif
        //possible optimization here
        while (col_count)
        {
            q7_t a0 = *ip_a0++;
            q15_t b0 = *ip_b0++;
            q15_t b1 = *ip_b1++;

            ch_0_out_0 += a0 * b0;
            ch_0_out_1 += a0 * b1;
            col_count--;
        }
        ch_0_out_0 = muriscv_nn_requantize(ch_0_out_0, *out_mult, *out_shift);
        ch_0_out_0 += out_offset;
        ch_0_out_0 = MAX(ch_0_out_0, activation_min);
        ch_0_out_0 = MIN(ch_0_out_0, activation_max);
        *out_0++ = (q7_t)ch_0_out_0;

        ch_0_out_1 = muriscv_nn_requantize(ch_0_out_1, *out_mult, *out_shift);
        ch_0_out_1 += out_offset;
        ch_0_out_1 = MAX(ch_0_out_1, activation_min);
        ch_0_out_1 = MIN(ch_0_out_1, activation_max);
        *out_1++ = (q7_t)ch_0_out_1;
        out_mult++;
        out_shift++;
        remaining--;
    }

    out_0 += output_ch;

    /* return the new output pointer with offset */
    return out_0;
#else
    (void)input_a;
    (void)input_b;
    (void)output_ch;
    (void)out_shift;
    (void)out_mult;
    (void)out_offset;
    (void)activation_min;
    (void)activation_max;
    (void)num_col_a;
    (void)output_bias;
    (void)out_0;
    /* To be completed */
    return NULL;
#endif




}

muriscv_nn_status muriscv_nn_convolve_s8(const muriscv_nn_context *ctx,
                                         const muriscv_nn_conv_params *conv_params,
                                         const muriscv_nn_per_channel_quant_params *quant_params,
                                         const muriscv_nn_dims *input_dims,
                                         const q7_t *input_data,
                                         const muriscv_nn_dims *filter_dims,
                                         const q7_t *filter_data,
                                         const muriscv_nn_dims *bias_dims,
                                         const int32_t *bias_data,
                                         const muriscv_nn_dims *output_dims,
                                         q7_t *output_data)
{
  (void)bias_dims;

    if (ctx->buf == NULL && muriscv_nn_convolve_s8_get_buffer_size(input_dims, filter_dims) > 0)
    {
        return MURISCV_NN_ARG_ERROR;
    }
    q15_t *buffer_a = (q15_t *)ctx->buf;

    const int32_t input_batches = input_dims->n;
    const uint16_t input_x = input_dims->w;
    const uint16_t input_y = input_dims->h;
    const uint16_t input_ch = input_dims->c;
    const uint16_t kernel_x = filter_dims->w;
    const uint16_t kernel_y = filter_dims->h;
    const uint16_t output_x = output_dims->w;
    const uint16_t output_y = output_dims->h;
    const uint16_t output_ch = output_dims->c;

    const uint16_t pad_x = conv_params->padding.w;
    const uint16_t pad_y = conv_params->padding.h;
    const uint16_t stride_x = conv_params->stride.w;
    const uint16_t stride_y = conv_params->stride.h;

    const int32_t input_offset = conv_params->input_offset;
    const int32_t out_offset = conv_params->output_offset;
    const int32_t out_activation_min = conv_params->activation.min;
    const int32_t out_activation_max = conv_params->activation.max;
    int32_t *output_mult = quant_params->multiplier;
    int32_t *output_shift = quant_params->shift;
    for (int i_batch = 0; i_batch < input_batches; i_batch++)
    {
      const uint16_t dilation_x = conv_params->dilation.w;
      const uint16_t dilation_y = conv_params->dilation.h;

      int32_t i_out_y, i_out_x, i_ker_y, i_ker_x;

      /* Generate two columns from the input tensor a GEMM computation */
        q15_t *two_column_buf = buffer_a;
        q7_t *out = output_data;

        /* This part implements the im2col function */
        for (i_out_y = 0; i_out_y < output_y; i_out_y++)
        {
            for (i_out_x = 0; i_out_x < output_x; i_out_x++)
            {
                const int32_t base_idx_y = stride_y * i_out_y - pad_y;
                const int32_t base_idx_x = stride_x * i_out_x - pad_x;

                for (i_ker_y = 0; i_ker_y < kernel_y; i_ker_y++)
                {
                    for (i_ker_x = 0; i_ker_x < kernel_x; i_ker_x++)
                    {
                        const int32_t k_y = base_idx_y + dilation_y * i_ker_y;
                        const int32_t k_x = base_idx_x + dilation_x * i_ker_x;

                        if (k_y < 0 || k_y >= input_y || k_x < 0 || k_x >= input_x)
                        {
                            /* Filling 0 for out-of-bound paddings */
                            memset((int8_t *)two_column_buf, 0, sizeof(q15_t) * input_ch);
                        }
                        else
                        {
                            /* Copying the pixel data to column */
                            muriscv_nn_q7_to_q15_with_offset(
                                input_data + (k_y * input_x + k_x) * input_ch, two_column_buf, input_ch, input_offset);
                        }
                        two_column_buf += input_ch;
                    }
                }

                /* Computation is filed for every 2 columns */
                if (two_column_buf == buffer_a + 2 * input_ch * kernel_y * kernel_x)
                {
                    out = muriscv_nn_mat_mult_kernel_s8_s16(filter_data,
                                                            buffer_a,
                                                            output_ch,
                                                            output_shift,
                                                            output_mult,
                                                            out_offset,
                                                            out_activation_min,
                                                            out_activation_max,
                                                            input_ch * kernel_y * kernel_x,
                                                            bias_data,
                                                            out);

                    /* counter reset */
                    two_column_buf = buffer_a;
                }
            }
        }

        /* left-over because odd number of output pixels */
        if (two_column_buf != buffer_a)
        {
            const q7_t *ker_a = filter_data;
            int i;

            for (i = 0; i < output_ch; i++)
            {
                /* Load the accumulator with bias first */
                q31_t sum = 0;
                if (bias_data)
                {
                    sum = bias_data[i];
                }

                /* Point to the beginning of the im2col buffer where the input is available as a rearranged column */
                const q15_t *ip_as_col = buffer_a;

/* 4 multiply and accumulates are done in one loop. */
#if defined(USE_PEXT)
                
                uint16_t col_count = (input_ch * kernel_y * kernel_x) >> 2;
                
                
                while (col_count)
                {
                    q31_t inA = muriscv_nn_read_q7x4_ia_fast(&ker_a);

                    q31_t ker_a1 = __rv__sunpkd810(inA);
                    q31_t ker_a2 = __rv__sunpkd832(inA);

                    q31_t ip_b1 = muriscv_nn_read_q15x2_ia_fast(&ip_as_col);
                    sum = __rv__kmada(sum, ker_a1, ip_b1);

                    q31_t ip_b2 = muriscv_nn_read_q15x2_ia_fast(&ip_as_col);
                    sum = __rv__kmada(sum, ker_a2, ip_b2);

                    col_count--;
            
                }
                /* Handle left over mac */
                col_count = input_ch * kernel_y * kernel_x & 0x3;
#else  /* defined(USE_PEXT) */
                uint16_t col_count = input_ch * kernel_y * kernel_x;
#endif /* defined(USE_PEXT) */
                while (col_count)
                {
                    q7_t ker_a1 = *ker_a++;
                    q15_t ip_b1 = *ip_as_col++;
                    sum += ker_a1 * ip_b1;
                    col_count--;
                }

                sum = muriscv_nn_requantize(sum, output_mult[i], output_shift[i]);
                sum += out_offset;
                sum = MAX(sum, out_activation_min);
                sum = MIN(sum, out_activation_max);
                *out++ = (q7_t)sum;
            }
        }

        /* Advance to the next batch */
        input_data += (input_x * input_y * input_ch);
        output_data += (output_x * output_y * output_ch);
    }
    return MURISCV_NN_SUCCESS;
}

namespace tflite {
namespace {

struct OpData {
  OpDataConv reference_op_data;

  // Index to buffer for optimizations if applicable.
  int buffer_idx;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  int32_t buf_size = 0;
  const auto& params =
      *(static_cast<const TfLiteConvParams*>(node->builtin_data));
  OpData* data = static_cast<OpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempOutputTensor(node, kConvBiasTensor);
  TfLiteType bias_type = bias != nullptr ? bias->type : kTfLiteNoType;

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context,
                     input->type == kTfLiteFloat32 ||
                         input->type == kTfLiteInt16 ||
                         input->type == kTfLiteInt8,
                     "Input data type not supported");
  TF_LITE_ENSURE_MSG(
      context,
      (input->type == kTfLiteFloat32 && filter->type == kTfLiteFloat32) ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 &&
           (filter->type == kTfLiteInt4 || filter->type == kTfLiteInt8)),
      "Hybrid models are not supported on TFLite Micro.");

  // Consistency check tensor dims
  // Dimensionality
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, output->dims->size, 4);
  // Equal batch size in input and output
  TF_LITE_ENSURE_EQ(context, input->dims->data[0], output->dims->data[0]);
  // Input channels should be an even multiple of filter channels
  TF_LITE_ENSURE(context, filter->dims->data[3] > 0);
  TF_LITE_ENSURE_EQ(context, input->dims->data[3] % filter->dims->data[3], 0);
  // Output channels should be an even multiple of the number of groups
  const int groups = input->dims->data[3] / filter->dims->data[3];
  TFLITE_DCHECK_EQ(output->dims->data[3] % groups, 0);
  // Bias size equal to output channels
  if (bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, bias->dims->size, 4);
    const int bias_size = NumElements(bias->dims);
    TFLITE_DCHECK_EQ(bias_size, output->dims->data[3]);
  }

  // Initialize muriscv_nn dimensions
  muriscv_nn_dims input_dims;
  input_dims.n = input->dims->data[0];
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input->dims->data[3];

  muriscv_nn_dims filter_dims;
  filter_dims.n = 1;
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = filter->dims->data[3];

  muriscv_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output->dims->data[3];

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    const int num_channels = filter->dims->data[kConvQuantizedDimension];
    data->reference_op_data.per_channel_output_multiplier =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, num_channels * sizeof(int32_t)));
    data->reference_op_data.per_channel_output_shift =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, num_channels * sizeof(int32_t)));
  }

  TF_LITE_ENSURE_STATUS(CalculateOpDataConv(
      context, node, params, input_dims.w, input_dims.h, filter_dims.w,
      filter_dims.h, output_dims.w, output_dims.h, input->type,
      &data->reference_op_data));

  // muriscv_NN allows INT64 or nullptr bias data pointer
  if (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 &&
       (bias_type == kTfLiteInt64 || bias_type == kTfLiteNoType))) {
    // Initialize muriscv_nn convolution parameters
    muriscv_nn_conv_params conv_params;
    conv_params.input_offset = -input->params.zero_point;
    conv_params.output_offset = output->params.zero_point;
    conv_params.stride.h = params.stride_height;
    conv_params.stride.w = params.stride_width;
    conv_params.dilation.h = params.dilation_height_factor;
    conv_params.dilation.w = params.dilation_width_factor;
    conv_params.padding.h = data->reference_op_data.padding.height;
    conv_params.padding.w = data->reference_op_data.padding.width;
    conv_params.activation.min = data->reference_op_data.output_activation_min;
    conv_params.activation.max = data->reference_op_data.output_activation_max;

    if (input->type == kTfLiteInt8) {
      buf_size = muriscv_nn_convolve_s8_get_buffer_size(
          &input_dims, &filter_dims);
    } 
    // else if (input->type == kTfLiteInt16) {
    //   TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    //   TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    //   buf_size = arm_convolve_wrapper_s16_get_buffer_size(
    //       &conv_params, &input_dims, &filter_dims, &output_dims);
    // }

    if (buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, buf_size, &data->buffer_idx));
    } else {
      data->buffer_idx = -1;
    }
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }

  return kTfLiteOk;
}

template <class ActType, class BiasType, class WeigthsType>
muriscv_nn_status convolve_wrapper(
    const muriscv_nn_context* ctx, const muriscv_nn_conv_params* conv_params,
    const muriscv_nn_per_channel_quant_params* quant_params,
    const muriscv_nn_dims* input_dims, const ActType* input,
    const muriscv_nn_dims* filter_dims, const int8_t* filter,
    const muriscv_nn_dims* bias_dims, const BiasType* bias,
    const muriscv_nn_dims* output_dims, ActType* output, WeigthsType weightsT) {
  return MURISCV_NN_ARG_ERROR;
}

template <>
muriscv_nn_status convolve_wrapper(
    const muriscv_nn_context* ctx, const muriscv_nn_conv_params* conv_params,
    const muriscv_nn_per_channel_quant_params* quant_params,
    const muriscv_nn_dims* input_dims, const int8_t* input,
    const muriscv_nn_dims* filter_dims, const int8_t* filter,
    const muriscv_nn_dims* bias_dims, const int32_t* bias,
    const muriscv_nn_dims* output_dims, int8_t* output, TfLiteType weightsT) {
  if (weightsT == kTfLiteInt8) {
    return muriscv_nn_convolve_s8(ctx, conv_params, quant_params, input_dims,
                                   input, filter_dims, filter, bias_dims, bias,
                                   output_dims, output);
  } 
  // else if (weightsT == kTfLiteInt4) {
  //   return arm_convolve_wrapper_s4(ctx, conv_params, quant_params, input_dims,
  //                                  input, filter_dims, filter, bias_dims, bias,
  //                                  output_dims, output);
  // }
   else {
    return MURISCV_NN_ARG_ERROR;
  }
}

// template <>
// arm_cmsis_nn_status convolve_wrapper(
//     const muriscv_nn_context* ctx, const muriscv_nn_conv_params* conv_params,
//     const muriscv_nn_per_channel_quant_params* quant_params,
//     const muriscv_nn_dims* input_dims, const int16_t* input,
//     const muriscv_nn_dims* filter_dims, const int8_t* filter,
//     const muriscv_nn_dims* bias_dims, const int64_t* bias,
//     const muriscv_nn_dims* output_dims, int16_t* output, TfLiteType weightsT) {
//   const muriscv_nn_bias_data bias_data = {bias, false};

//   return arm_convolve_wrapper_s16(ctx, conv_params, quant_params, input_dims,
//                                   input, filter_dims, filter, bias_dims,
//                                   &bias_data, output_dims, output);
// }

// template <>
// arm_cmsis_nn_status convolve_wrapper(
//     const muriscv_nn_context* ctx, const muriscv_nn_conv_params* conv_params,
//     const muriscv_nn_per_channel_quant_params* quant_params,
//     const muriscv_nn_dims* input_dims, const int16_t* input,
//     const muriscv_nn_dims* filter_dims, const int8_t* filter,
//     const muriscv_nn_dims* bias_dims, const int32_t* bias,
//     const muriscv_nn_dims* output_dims, int16_t* output, TfLiteType weightsT) {
//   const muriscv_nn_bias_data bias_data = {bias, true};

//   return arm_convolve_wrapper_s16(ctx, conv_params, quant_params, input_dims,
//                                   input, filter_dims, filter, bias_dims,
//                                   &bias_data, output_dims, output);
// }

template <typename ActType, typename BiasType, TfLiteType type>
TfLiteStatus EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                                     const TfLiteConvParams& params,
                                     const OpData& data,
                                     const TfLiteEvalTensor* input,
                                     const TfLiteEvalTensor* filter,
                                     const TfLiteEvalTensor* bias,
                                     TfLiteEvalTensor* output) {
  muriscv_nn_conv_params conv_params;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;

  // Initialize muriscv_nn convolution parameters
  conv_params.input_offset = -data.reference_op_data.input_zero_point;
  conv_params.output_offset = data.reference_op_data.output_zero_point;
  conv_params.stride.h = params.stride_height;
  conv_params.stride.w = params.stride_width;
  conv_params.padding.h = data.reference_op_data.padding.height;
  conv_params.padding.w = data.reference_op_data.padding.width;
  conv_params.activation.min = data.reference_op_data.output_activation_min;
  conv_params.activation.max = data.reference_op_data.output_activation_max;

  // Initialize muriscv_nn per channel quantization parameters
  muriscv_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(
      data.reference_op_data.per_channel_output_multiplier);
  quant_params.shift =
      const_cast<int32_t*>(data.reference_op_data.per_channel_output_shift);

  // Initialize muriscv_nn dimension structs, consistency is checked in the
  // prepare stage
  muriscv_nn_dims input_dims;
  input_dims.n = input->dims->data[0];
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input->dims->data[3];

  muriscv_nn_dims filter_dims;
  filter_dims.n = 1;
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = filter->dims->data[3];

  muriscv_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output->dims->data[3];

  muriscv_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output->dims->data[3];

  // Initialize muriscv_nn context
  muriscv_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  if (data.buffer_idx > -1) {
    ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
    // Note: ctx.size is currently not used in muriscv_nn.
    // The buffer should be allocated in the prepare function through
    // the corresponding arm_convolve_wrapper_[type]_get_buffer_size
  }

  // arm_convolve_wrapper_[type] dispatches the optimized kernel accordingly
  // with the parameters passed
  TFLITE_DCHECK_EQ(
      convolve_wrapper(
          &ctx, &conv_params, &quant_params, &input_dims,
          tflite::micro::GetTensorData<ActType>(input), &filter_dims,
          tflite::micro::GetTensorData<int8_t>(filter), &bias_dims,
          tflite::micro::GetOptionalTensorData<BiasType>(bias), &output_dims,
          tflite::micro::GetTensorData<ActType>(output), type),
      MURISCV_NN_SUCCESS);

  return kTfLiteOk;
}

TfLiteStatus EvalInt4(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  return EvalQuantizedPerChannel<int8_t, int32_t, kTfLiteInt4>(
      context, node, params, data, input, filter, bias, output);
}

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  return EvalQuantizedPerChannel<int8_t, int32_t, kTfLiteInt8>(
      context, node, params, data, input, filter, bias, output);
}

TfLiteStatus EvalInt16x8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  if (bias == nullptr || bias->type == kTfLiteInt32) {
    return EvalQuantizedPerChannel<int16_t, int32_t, kTfLiteInt16>(
        context, node, params, data, input, filter, bias, output);
  } else if (bias->type == kTfLiteInt64) {
    return EvalQuantizedPerChannel<int16_t, int64_t, kTfLiteInt16>(
        context, node, params, data, input, filter, bias, output);
  } else {
    MicroPrintf("Bias type %s (%d) not supported.",
                TfLiteTypeGetName(bias->type), bias->type);
    return kTfLiteError;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 && filter->type == kTfLiteInt4),
      "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data.reference_op_data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          return EvalQuantizedPerChannel<int8_t, int32_t, kTfLiteInt4>(
              context, node, params, data, input, filter, bias, output);
        }
        case kTfLiteInt8: {
          return EvalQuantizedPerChannel<int8_t, int32_t, kTfLiteInt8>(
              context, node, params, data, input, filter, bias, output);
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
        }
      }
      break;
    }
    case kTfLiteInt16: {
      if (bias == nullptr || bias->type == kTfLiteInt32) {
        return EvalQuantizedPerChannel<int16_t, int32_t, kTfLiteInt16>(
            context, node, params, data, input, filter, bias, output);
      } else if (bias->type == kTfLiteInt64) {
        return EvalQuantizedPerChannel<int16_t, int64_t, kTfLiteInt16>(
            context, node, params, data, input, filter, bias, output);
      } else {
        MicroPrintf("Bias type %s (%d) not supported.",
                    TfLiteTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

// TFLMRegistration Register_CONV_2D_INT4() {
//   return tflite::micro::RegisterOp(Init, Prepare, EvalInt4);
// }

// TFLMRegistration Register_CONV_2D_INT8() {
//   return tflite::micro::RegisterOp(Init, Prepare, EvalInt8);
// }

// TFLMRegistration Register_CONV_2D_INT16() {
//   return tflite::micro::RegisterOp(Init, Prepare, EvalInt16x8);
// }

}  // namespace tflite