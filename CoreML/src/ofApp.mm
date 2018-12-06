#include "ofApp.h"
#import <Accelerate/Accelerate.h>  // for matrix operation
#include <algorithm>               //

const float anchors[]={1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};
int inputWidth = 416;
int inputHeight = 416;
int  maxBoundingBoxes = 10;

const int c_color[20][3]={{51,77,102},{51,77,204},{51,179,102},{51,179,204},{102,77,102},{102,77,204},{102,179,102},{102,179,204},{153,77,102},{153,77,204},{153,179,102},{153,179,204}, {204,77,102},{204,77,204},{204,179,102},{204,179,204}, {255,77,102},{255,77,204},{255,179,102},{255,179,204}};

string labels[20] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

// Tweak these values to get more or fewer predictions.
float confidenceThreshold = 0.1;
float iouThreshold = 0.5;
float scale = 1.0;
float scaleX = 1.0;

struct Prediction {
    int classIndex;
    float score;
    CGRect rect;
};

Prediction predictions[30]; // 出力用のバウンディングボックス構造体の数この程度あれば十分　下処理用
Prediction selected[30];    //
Prediction select_end[30];

TinyYOLO *yolo = [[TinyYOLO alloc] init];
TinyYOLOInput *yoloin =[TinyYOLOInput alloc];
TinyYOLOOutput *yoloout =[TinyYOLOOutput alloc];

MLMultiArray *grids;
NSError *labelx=[NSError alloc];
float ttt,ttt1;
ofImage img,img1;
cv::Mat mat;
int selected_count=0;
int count_end=0;
int class_count=0;
//*************************************
CVPixelBufferRef yorox(cv::Mat mat2);
float sigmoid(float sig);
float IOU(CGRect a, CGRect b);
void nonMaxSuppression(Prediction *boxes, int limt, float threshold,int count);
void computeBoundingBoxes(cv::Mat mat1);

//  multi-thread DetectNet
class Detect_x: public ofThread {
public:
    void threadedFunction(){
        
        ok=false;
        ttt=ofGetElapsedTimef();
        computeBoundingBoxes(mat);   //
        
        lock();
        nonMaxSuppression(predictions, maxBoundingBoxes, iouThreshold,class_count);
        unlock();
        ok=true;
        ttt=ofGetElapsedTimef()-ttt;  // 推測処理の時間計測
        ttt1=ttt;
    }
    bool ok;
};
Detect_x Found_X;
//*************************************

//------------imgをCVPixelBufferRefに変換　ofxCVを使用-------------------------------
CVPixelBufferRef yorox(cv::Mat mat2)
{
    int width = mat2.cols;
    int height = mat2.rows;
yoyo:
    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                             [NSNumber numberWithInt:width], kCVPixelBufferWidthKey,
                             [NSNumber numberWithInt:height], kCVPixelBufferHeightKey,
                             nil];
    CVPixelBufferRef imageBuffer=Nil;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorMalloc, width, height, kCVPixelFormatType_32BGRA, (CFDictionaryRef) CFBridgingRetain(options), &imageBuffer) ;
    if(status != kCVReturnSuccess && imageBuffer == NULL) goto yoyo; //こうなったらエラー再設定
    
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    void *base = CVPixelBufferGetBaseAddress(imageBuffer) ;
    memcpy(base, mat2.data, mat2.total()*4);
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
    return imageBuffer;
}

float sigmoid(float sig) {
    return 1.0f / (1.0f + exp(-sig));
}

float IOU(CGRect a, CGRect b){
    float areaA = a.size.width * a.size.height;
    if (areaA <= 0) { return 0 ;}
    
    float areaB = b.size.width * b.size.height;
    if (areaB <= 0) { return 0; }
    CGFloat  intersectionMinX = std::max(CGRectGetMinX(a), CGRectGetMinX(b));
    CGFloat  intersectionMinY = std::max(CGRectGetMinY(a), CGRectGetMinY(b));
    CGFloat  intersectionMaxX = std::min(CGRectGetMaxX(a), CGRectGetMaxX(b));
    CGFloat  intersectionMaxY = std::min(CGRectGetMaxY(a), CGRectGetMaxY(b));
    CGFloat  intersectionArea = std::max(intersectionMaxY - intersectionMinY, 0.0) * std::max(intersectionMaxX - intersectionMinX, 0.0);
    return (float)intersectionArea / (float)(areaA + areaB - intersectionArea);
}

void nonMaxSuppression(Prediction *boxes, int limt, float threshold,int count){
    vDSP_Length sortedIndices[]={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};  //最終的なsort用のダミーインデックス
    bool active []={true,true,true,true,true,true,true,true,true,true,true,true,true,true,true,};//重複削除用に使用
    selected_count=0;
    // Do an argsort on the confidence scores, from high to low.
    float s_data[count];
    for(int i=0;i<count;i++){s_data[i]=boxes[i].score;}
    vDSP_vsorti(s_data, sortedIndices, NULL, count, -1);  //sortIndeicesにでーたでそーとしたインデックス番号が入るはずなんだけどうまくいってる！！ -1が降順　1が昇順
    int numActive=count;
    // The algorithm is simple: Start with the box that has the highest score.
    // Remove any remaining boxes that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
outer: for (int i=0;i<count;i++) {
    if (active[i]) {
        Prediction boxA = boxes[sortedIndices[i]];
        selected[i]=boxA;selected_count++;
        if (selected_count >= limt)  break;
        for (int j=i+1;j<count;j++)  {
            if( active[j]) {
                Prediction boxB = boxes[sortedIndices[j]];
                if (IOU(boxA.rect, boxB.rect) > threshold) {
                    active[j] = false;
                    numActive -= 1;
                    if (numActive <= 0) {goto outer;}
                }
            }
        }
    }
}
    //  printf("Aho2\n");
}
//--------------------------------------------------------------
void computeBoundingBoxes(cv::Mat mat1) {
    yoloout=[yolo predictionFromImage:yorox(mat1) error:NULL]; // send image to CoreML and receive result
    MLMultiArray *features=yoloout.grid;
    assert(features.count == 125*13*13);
    float   blockSize = 32.0f;
    int     gridHeight = 13;
    int     gridWidth = 13;
    int     boxesPerCell = 5;
    int     numClasses = 20;
    
    int channelStride = features.strides[0].intValue;
    int yStride =       features.strides[1].intValue;
    int xStride =       features.strides[2].intValue;
    class_count=0;
    
    // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
    // five data items: x, y, width, height, and a confidence score. Each grid
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell
    // values for each grid cell, i.e. 125 channels. The total features array
    // contains 125x13x13 elements.
    
    // NOTE: It turns out that accessing the elements in the multi-array as
    // `features[[channel, cy, cx] as [NSNumber]].floatValue` is kinda slow.
    // It's much faster to use direct memory access to the features.
    double *featurePointer=static_cast<double*>(features.dataPointer);
    
    for (int cy=0;cy<gridHeight;cy++) {
        for(int cx=0;cx<gridWidth;cx++) {
            for(int b= 0;b<boxesPerCell;b++) {
                // For the first bounding box (b=0) we have to read channels 0-24,
                // for b=1 we have to read channels 25-49, and so on.
                int channel = b*(numClasses + 5);
                int cxy=cx*xStride+cy*yStride;
                
                float tx = featurePointer[channel      *channelStride+cxy];
                float ty = featurePointer[(channel + 1)*channelStride+cxy];
                float tw = featurePointer[(channel + 2)*channelStride+cxy];
                float th = featurePointer[(channel + 3)*channelStride+cxy];
                float tc = featurePointer[(channel + 4)*channelStride+cxy];
                
                // The predicted tx and ty coordinates are relative to the location
                // of the grid cell; we use the logistic sigmoid to constrain these
                // coordinates to the range 0 - 1. Then we add the cell coordinates
                // (0-12) and multiply by the number of pixels per grid cell (32).
                // Now x and y represent center of the bounding box in the original
                // 416x416 image space.
                float x = ((float)cx + sigmoid(tx)) * blockSize;
                float y = ((float)cy + sigmoid(ty)) * blockSize;
                // The size of the bounding box, tw and th, is predicted relative to
                // the size of an "anchor" box. Here we also transform the width and
                // height into the original 416x416 image space.
                float w = exp(tw) * anchors[2*b    ] * blockSize;
                float h = exp(th) * anchors[2*b + 1] * blockSize;
                // The confidence value for the bounding box is given by tc. We use
                // the logistic sigmoid to turn this into a percentage.
                float confidence = sigmoid(tc);
                /*
                 Computes the "softmax"  over an array.
                 Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/
                 This is what softmax looks like in "pseudocode" (actually using Python
                 and numpy):
                 x -= np.max(x)
                 exp_scores = np.exp(x)
                 softmax = exp_scores / np.sum(exp_scores)
                 */
                float classes[numClasses];
                for(int c=0;c<numClasses;c++) {
                    classes[c] = featurePointer[(channel + 5 + c)*channelStride+cxy];
                }
                
                //今回の場合繰り返しが20回だけだが、最低3回のfor文＝60回ここで回す必要ある
                //ただしさらに13x13x5回、演算が必要なので、以下が少しでも早いと処理が有利になる
                //なので以下XCODEの行列演算ライブラリvDSPを使っている。確実に効果が出る。
                //ただし、#import <Accelerate/Accelerate.h>　が必要
                float max = 0;//結果は max＝配列の最大値
                vDSP_maxv(classes, 1, &max, 20);
                max = -max;//各値から最大値を引いて代入
                vDSP_vsadd(classes, 1, &max, classes, 1, 20);
                int count=20;//各値でexpを計算
                vvexpf(classes ,classes,&count);
                float sum=0;//合計をsumに代入して、各値/sumで再格納
                vDSP_sve(classes, 1, &sum, 20);
                vDSP_vsdiv(classes ,1, &sum, classes, 1, 20);
                vDSP_Length  class_x=0;
                float max_x=0;     //max_x に最大値を   class_x　に最大値の配列番号を入れる
                vDSP_maxvi(classes, 1, &max_x,&class_x,20);
                
                // Find the index of the class with the largest score.
                int detectedClass=      class_x;
                float bestClassScore =  max_x;
                
                // Combine the confidence score for the bounding box, which tells us
                // how likely it is that there is an object in this box (but not what
                // kind of object it is), with the largest class prediction, which
                // tells us what kind of object it detected (but not where).
                float confidenceInClass = bestClassScore * confidence;
                
                //printf("confidence %f\n",confidence);
                // Since we compute 13x13x5 = 845 bounding boxes, we only want to
                // keep the ones whose combined score is over a certain threshold.
                if (confidenceInClass > 0.3) {    //0.4  正解率！！
                    //  printf("classcount=%f\n",confidenceInClass);
                    CGRect rect = CGRectMake (CGFloat(x - w/2),  CGFloat(y - h/2),CGFloat(w), CGFloat(h));
                    predictions[class_count] = { detectedClass,confidenceInClass,rect};
                    if (class_count < maxBoundingBoxes) class_count++;
                    
                }
            }
        }
    }
}

//--------------------------------------------------------------
void ofApp::setup(){
    ofSetFrameRate(60);
    
#ifdef CAM
    grabber.setDeviceID(0);
    grabber.setup(640, 480);
#elif
    grabber.setup(640,640, true);
#endif

    img.allocate(grabber.getWidth(), grabber.getHeight(), OF_IMAGE_COLOR);
    Found_X.ok = true;
    scale = 1.0;//720.0/416.0;
    scaleX = scale;// * (16.0/9.0);//1;//(16.0/9.0);
}

//--------------------------------------------------------------
void ofApp::update(){
//    ofSetFrameRate((int)1/ttt1);   //シンクロしないので無理栗フレームレートを変化させる - error on macos

    ofBackground(255, 255, 255);
#ifdef CAM
    grabber.update();
    if (grabber.isFrameNew() == true) {
        img.setFromPixels(grabber.getPixels().getData(), grabber.getWidth(), grabber.getHeight(), OF_IMAGE_COLOR);
        img.crop(0, 0, 480, 480);
        img.resize(416, 416);

        if (Found_X.ok) {
            Found_X.ok = false;
            mat = ofxCv::toCv(img);
            cv::cvtColor(mat, mat, CV_BGR2BGRA);
            Found_X.startThread();
        }
    }

#elif
    grabber.grabScreen(160, 160);
    nonMaxSuppression(predictions, maxBoundingBoxes, iouThreshold, class_count);
    ofTexture t = grabber.getTextureReference();
    ofPixels p;
    t.readToPixels(p);
    img.setFromPixels(p);//, 640,480, OF_IMAGE_COLOR);
    img.crop(0, 0, 640, 640);
    img.resize(416, 416);

    if (Found_X.ok) {
        Found_X.ok = false;
        mat = ofxCv::toCv(img);
        cv::cvtColor(mat, mat, CV_BGR2BGRA);
        Found_X.startThread();
    }
#endif
}

//--------------------------------------------------------------
void ofApp::draw(){
    ofBackground(0, 0, 0);
    ofSetColor(255);
    img.draw(0,0,416*scaleX,416*scale);
    //grabber.draw(0, 0, 640, 480);
    // printf("selected_count%i\n",selected_count);
    for(int i=0;i<selected_count;i++){     //selected_count:ボックスの数　selected:ボックスの構造体
        
        //クラス範囲を示す四角形を書く　レーティーナ対応のため画像を1.5倍しているので、四角形も1.5倍
        ofNoFill();
        ofSetLineWidth(1);
        ofSetColor(c_color[selected[i].classIndex][0],c_color[selected[i].classIndex][1],c_color[selected[i].classIndex][2]);
    ofDrawRectangle(selected[i].rect.origin.x*scaleX,selected[i].rect.origin.y*scale,selected[i].rect.size.width*scaleX,selected[i].rect.size.height*scale);
        stringstream ss;  //Box左上にクラス名称を表示
        ss << labels[selected[i].classIndex] <<" "<<ofToString(selected[i].score*100,1) << "\n";
        ofFill();
        ofDrawRectangle(selected[i].rect.origin.x*scaleX,selected[i].rect.origin.y*scale+2,ss.str().length()*8.5,15);
        ofSetColor(255);
        ofDrawBitmapString(ss.str().c_str(), selected[i].rect.origin.x*scaleX + 5,selected[i].rect.origin.y*scale + 12);
        // ofFill();  //Boxに半透明の色塗り
        ofSetColor(c_color[selected[i].classIndex][0],c_color[selected[i].classIndex][1],c_color[selected[i].classIndex][2],60);
    ofDrawRectangle(selected[i].rect.origin.x*scaleX,selected[i].rect.origin.y*scale,selected[i].rect.size.width*scaleX,selected[i].rect.size.height*scale);
        ofSetColor(255);
        
        //認識した部分のイメージを下部に描画する
//        ofDisableAlphaBlending();
//        img1.cropFrom(img,selected[i].rect.origin.x,selected[i].rect.origin.y,selected[i].rect.size.width,selected[i].rect.size.height);
//        img.clear();
//        img1.draw(100,416*1.8);
//        ofEnableAlphaBlending();
//        img1.clear();
        
    }
    //実際の認識で消費した時間を描画する　iPhone8=0.065程度　15FPS前後　iPhone X 0.045程度　　22FPS前後
    ofSetColor(255);
    stringstream ss2;
    ss2 << "Recognition: " << ofToString(ttt1,4) <<" seconds"<< "\n";
    ofDrawBitmapString(ss2.str().c_str(), 20, 416*scale+20);
    
    string info;
    info +="FPS    : " + ofToString(ofGetFrameRate(),2)+"   ";
    ofDrawBitmapString(info, 20, 416*scale+40);

}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
