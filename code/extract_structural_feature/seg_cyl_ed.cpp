#include <vector>   //可以跑下垂圆柱分割、等距分割
#include <pcl/point_types.h>  //
#include <pcl/io/pcd_io.h>  //点云数据输入输出模块
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/cloud_viewer.h>  //点云显示
#include <pcl/visualization/pcl_visualizer.h>  //点云显示
#include <boost/thread/thread.hpp>
#include <pcl/sample_consensus/method_types.h>   //随机参数估计方法头文件
#include <pcl/sample_consensus/model_types.h>    //模型定义头文件
#include <pcl/segmentation/sac_segmentation.h>   //基于采样一致性分割的类的头文件
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>		//直通滤波
#include <pcl/filters/statistical_outlier_removal.h>  //统计滤波去除离群点
#include <cmath> 
#include <iostream>
#include <iomanip>
#include <pcl/surface/gp3.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/common/common.h>
#include <fstream> 
#include <math.h> 
#include <pcl/common/distances.h>
using namespace std;
typedef pcl::PointXYZ PointT;

#define Fb1 0.02  //设置的分割间隔，后面分割间隔 H 和 H0 直接用Fb1赋值，就方便了
#define Fb2 0.0008//设置截取叶片点云的宽度
#define Fb3 0.02 //等距分割的距离间隔
int pm = 2;  //1选择重心点  2选择几何中心点
size_t time_id = 14; //第几次采集
size_t group_id = 4; //第几组
size_t plant_id = 6; //第几株
size_t leaf_id = 10; //第几片叶片
size_t j = leaf_id;

//生成实现读写功能的对象
pcl::PCDWriter writer;
pcl::PCDReader reader;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("33D Viewer1"));
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer2(new pcl::visualization::PCLVisualizer("33D Viewer2"));
//定义圆柱分割的叶脉点云对象vein1(等距分割的为vein2)
pcl::PointCloud<pcl::PointXYZ>::Ptr vein1(new pcl::PointCloud<pcl::PointXYZ>());
pcl::PointCloud<pcl::PointXYZ>::Ptr vein2(new pcl::PointCloud<pcl::PointXYZ>());

int a[14][3] = { { 255, 0, 0 }, { 255, 128, 0 }, { 255,255,0 }, { 0,255,0 }, { 0,255,255 }, { 0,0,255 }, { 128,0,255 },
				 { 255, 0, 0 }, { 255, 128, 0 }, { 255,255,0 }, { 0,255,0 }, { 0,255,255 }, { 0,0,255 }, { 128,0,255 } };

//声明两个分割的函数
void cylinder_seg(pcl::PointXYZ center1,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_l_cly,
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_cly_1,
	int BS_cylinder, int* bs_cylinder, float k, bool k_is_valid, int j);
void equal_seg(float A, float B, float C, float D, int bs_eq,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_l_equ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_equ_1, int j);

int main(int argc, char** argv)
{
	viewer1->setBackgroundColor(0, 0, 0);
	viewer2->setBackgroundColor(0, 0, 0);

	//定义一系列点云对象   
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_stem(new pcl::PointCloud<pcl::PointXYZ>());  // cloud_stem茎秆  
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a_point(new pcl::PointCloud<pcl::PointXYZ>());  // 茎秆点云上那一点
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_leaf(new pcl::PointCloud<pcl::PointXYZ>());  //cloud_leaf叶片
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_cly_1(new pcl::PointCloud<pcl::PointXYZ>);//第一部分每个圆柱小片段点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_cly_2(new pcl::PointCloud<pcl::PointXYZ>);//第二部分每个圆柱小片段点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_equ_1(new pcl::PointCloud<pcl::PointXYZ>);//每个等距分割片段点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_equ_2(new pcl::PointCloud<pcl::PointXYZ>);//再次导入每隔一个等距截面曲线点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr veinfang(new pcl::PointCloud<pcl::PointXYZ>());	//叶脉方向向量点云veinfang，把叶脉点两两组成的方向向量按点云的方式存储起来

	float leaf_len_array[100]; //叶长数组
	int leaf_len_i = 0;       //小段叶片长序号     
	float leaf_len = 0;      //叶片总长初值
	int vein_size1 = 0; //圆柱分割片段叶脉点点云的个数
	int vein_size2 = 0; //等距截面曲线叶脉点点云的个数

	pcl::PointXYZ fang;//叶脉方向向量三个坐标值
	Eigen::Vector4d vec1(fang.x, fang.y, fang.z, 0);//定义的四维的叶脉方向向量，方便计算空间长度

	//导入茎秆点云
	std::stringstream  ss0; //导入茎秆点云字符串
	ss0 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-0" << ".pcd";
	if (!reader.read("E:/develop/pcl_lri/data/20250521/origin/" + ss0.str(), *cloud_stem))
	{
		//显示茎秆点云
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color19(cloud_stem, 255, 255, 0);
		viewer1->addPointCloud<pcl::PointXYZ>(cloud_stem, single_color19, "dd");
		//pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color20(cloud_stem, 255, 255, 0);
		//viewer2->addPointCloud<pcl::PointXYZ>(cloud_stem, single_color20, "dd");

		//这个地方对茎秆点云使用getMinMax3D，为了找那个圆柱上的一个中心点的坐标x和坐标y
		pcl::PointXYZ mind0, maxd0, center1;
		pcl::getMinMax3D(*cloud_stem, mind0, maxd0);
		center1.x = (maxd0.x + mind0.x) / 2;
		center1.y = (maxd0.y + mind0.y) / 2;

		//导入叶片点云
		for (int j = leaf_id; j <= leaf_id; j++) {
			std::stringstream ss1;  //导入叶片点云的字符串
			ss1 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-" << j << ".pcd";
			if (!reader.read("E:/develop/pcl_lri/data/20250521/origin/" + ss1.str(), *cloud_leaf))
			{
				// 创建一个KD树，用来实现mls，曲面重建
				pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
				// 输出文件中有PointNormal类型，用来存储移动最小二乘法算出的法线
				pcl::PointCloud<pcl::PointNormal> mls_points;
				pcl::PointCloud<pcl::PointNormal>::Ptr mls_points_normal(new pcl::PointCloud<pcl::PointNormal>);

				//将mls优化后的叶片点云mls_points_normal导入到cloud_l_cly 和cloud_l_equ，方便后续运算
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_l_cly(new pcl::PointCloud<pcl::PointXYZ>());
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_l_equ(new pcl::PointCloud<pcl::PointXYZ>());

				*cloud_l_cly = *cloud_leaf;
				*cloud_l_equ = *cloud_leaf;

				//这个地方获取了叶片点云，为了分类，分阈值lz<lxy 就是展开叶片,否则就是未展开叶片
				pcl::PointXYZ mind1, maxd1;
				pcl::getMinMax3D(*cloud_leaf, mind1, maxd1);
				float lz = 0, lxy = 0, lxz = 0;
				lz = maxd1.z - mind1.z;
				lxy = sqrt(pow((maxd1.x - mind1.x), 2) + pow((maxd1.y - mind1.y), 2));

				//找出那一个点坐标z，将这一点center1存入cloud_a_point点中，方便显示
				center1.z = mind1.z - 0.01;
				cloud_a_point->points.push_back(center1);
				//显示那个一点cloud_a_point，好像不设置点云宽度、高度、稠密也行
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> point_color(cloud_a_point, 255, 0, 0);
				viewer1->addPointCloud<pcl::PointXYZ>(cloud_a_point, point_color, "point_color");
				viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "point_color");
				viewer2->addPointCloud<pcl::PointXYZ>(cloud_a_point, point_color, "point_color");
				viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "point_color");

				//遍历寻找xoy投影面上的最远点坐标
				float xymax = -100;
				int flagxy = 0;
				for (int i = 0; i < cloud_l_cly->points.size(); i++) {
					if (sqrt(pow(cloud_l_cly->points[i].x - center1.x, 2) + pow(cloud_l_cly->points[i].y - center1.y, 2)) > xymax) {
						xymax = sqrt(pow(cloud_l_cly->points[i].x - center1.x, 2) + pow(cloud_l_cly->points[i].y - center1.y, 2));
						flagxy = i;
					}
				}
				//第二种方法，找z坐标最高点
				float zmax = -100;
				int flagz = 0;
				for (int i = 0; i < cloud_l_cly->points.size(); i++) {
					if (cloud_l_cly->points[i].z > zmax) {
						zmax = cloud_l_cly->points[i].z;
						flagz = i;
					}
				}

				// center1 在 xoy 面投影距离将叶片分成两部分并分别保存 
				double thresh = sqrt(
					pow(static_cast<double>(cloud_l_cly->points[flagz].x - center1.x), 2) +
					pow(static_cast<double>(cloud_l_cly->points[flagz].y - center1.y), 2)
				);

				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_part1(new pcl::PointCloud<pcl::PointXYZ>());
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_part2(new pcl::PointCloud<pcl::PointXYZ>());

				for (size_t idx = 0; idx < cloud_l_cly->points.size(); ++idx) {
					const auto& p = cloud_l_cly->points[idx];
					double d = sqrt(pow(static_cast<double>(p.x - center1.x), 2) + pow(static_cast<double>(p.y - center1.y), 2));
					if (d <= thresh) cloud_part1->points.push_back(p);
					else cloud_part2->points.push_back(p);
				}

				cloud_part1->width = static_cast<uint32_t>(cloud_part1->points.size());
				cloud_part1->height = 1;
				cloud_part1->is_dense = false;
				cloud_part2->width = static_cast<uint32_t>(cloud_part2->points.size());
				cloud_part2->height = 1;
				cloud_part2->is_dense = false;

				std::stringstream  ss3;
				ss3 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-" << leaf_id;
				const std::string file1 = "E:/develop/pcl_lri/data/20250521/cylinder - fuben2/" + ss3.str() + "-111.pcd";
				const std::string file2 = "E:/develop/pcl_lri/data/20250521/cylinder - fuben2/" + ss3.str() + "-222.pcd";
				writer.write<pcl::PointXYZ>(file1, *cloud_part1);
				writer.write<pcl::PointXYZ>(file2, *cloud_part2);
				std::cout << "Saved split files: " << file1 << " (" << cloud_part1->points.size() << " pts), "
					<< file2 << " (" << cloud_part2->points.size() << " pts)\n";

				//计算叶片生长方向斜率k
				float dx = cloud_l_cly->points[flagxy].x - center1.x;
				float dy = cloud_l_cly->points[flagxy].y - center1.y;
				const float EPS_K = 1e-8;
				float k = 0.0;
				bool k_is_valid = true;
				if (abs(dx) < EPS_K) {
					k_is_valid = false; //垂直线，斜率无穷大，不使用斜率表示
				}
				else {
					k = dy / dx;
					if (abs(k) < EPS_K) { k = 0.0; }
				}
				cout << "茎秆中心点(" << center1.x << "," << center1.y << "," << center1.z << ")" << endl;//输出圆柱上的那个点
				cout << "xymax坐标(" << cloud_l_cly->points[flagxy].x << "," << cloud_l_cly->points[flagxy].y << "," << cloud_l_cly->points[flagxy].z << ")" << endl;
				cout << "xymax的索引号：" << flagxy << endl;
				cout << "叶片生长方向：(1," << k << ")" << endl;
				if (k_is_valid) cout << "1," << k << ")";
				else cout << "vertical)";
				cout << endl;
				cout << "zmax坐标(" << cloud_l_cly->points[flagz].x << "," << cloud_l_cly->points[flagz].y << "," << cloud_l_cly->points[flagz].z << ")" << endl;
				cout << "zmax的索引号：" << flagz << endl << endl;


				int bs_cylinder = 0; //记录有点的圆柱分割的叶片片段序号
				//开始分别进行分割
				if (2 > 1) {
					for (int BS_cylinder = 1; BS_cylinder <= (lz / Fb1 + 60); BS_cylinder++)
					{
						//cylinder_seg();//进行圆柱分割
						cylinder_seg(center1, cloud_part1, ci_cly_1, BS_cylinder, &bs_cylinder, k, k_is_valid, j);

						//每个圆柱片段的显示ci_cly_1
						std::stringstream  ss2;
						ss2 << time_id << "oushi" << BS_cylinder;//增加独一无二的ID
						pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(ci_cly_1, a[bs_cylinder % 14][0], a[bs_cylinder % 14][1], a[bs_cylinder % 14][2]);
						viewer1->addPointCloud<pcl::PointXYZ>(ci_cly_1, single_color, ss2.str());
					}

					// 删除第一部分分割得到的最后一个叶脉点（误差较大），避免导入 vein1
					if (!vein1->points.empty()) {
						pcl::PointXYZ removed = vein1->points.back();
						vein1->points.pop_back();
						cout << "Removed last vein1 point from first-part segmentation: " << removed << endl;
					}

					//第二部分叶片的分割
					pcl::PointXYZ center2;
					center2.x = cloud_l_cly->points[flagz].x;
					center2.y = cloud_l_cly->points[flagz].y;
					center2.z = cloud_l_cly->points[flagz].z;
					for (int BS_cylinder = 1; BS_cylinder <= (lz / Fb1 + 50); BS_cylinder++) //对第二部分叶片进行圆柱分割
					{
						cylinder_seg(center2, cloud_part2, ci_cly_2, BS_cylinder, &bs_cylinder, k, k_is_valid, j);
						std::stringstream  ss10;
						ss10 << time_id << "part2" << BS_cylinder;//增加独一无二的ID
						pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(ci_cly_2, a[bs_cylinder % 14][0], a[bs_cylinder % 14][1], a[bs_cylinder % 14][2]);
						viewer1->addPointCloud<pcl::PointXYZ>(ci_cly_2, single_color, ss10.str());
					}
				}

				else {
					cout << "lz=" << lz << "  " << "lxy=" << lxy << "   lz<lxy 是未展开叶片" << endl;
				}

				//判断一下，多加入最末端的一个点为叶脉点
				//reader.read("E:/develop/pcl_angle/data/20250521/cylinder - fuben/" + ss1.str(), *ci_cly_2); 
				int tail_idx = 0;
				float dmax = -1.0f;
				// 找最后的ci_cly_2点云里面，距离倒数第二个叶脉点最远的点作为最后一个叶脉点
				if (cloud_part2->points.size() != 0)
				{
					for (size_t idx = 0; idx < ci_cly_2->points.size(); ++idx)
					{
						//float d = pcl::euclideanDistance(ci_cly_2->points[idx], cloud_l_equ->points[flagz]); //vein
						float d = pcl::euclideanDistance(ci_cly_2->points[idx], vein1->points[vein1->points.size() - 2]);
						if (d > dmax) {
							dmax = d;
							tail_idx = static_cast<int>(idx);
						}

					}
					// 与 vein1 的最后一个点比较，不同则加入
					const pcl::PointXYZ& candidate = ci_cly_2->points[tail_idx];
					const pcl::PointXYZ& last_vein1 = vein1->points.back();
					const float EPS = 1e-6f;
					if (fabs(candidate.x - last_vein1.x) > EPS || fabs(candidate.y - last_vein1.y) > EPS || fabs(candidate.z - last_vein1.z) > EPS) {
						vein1->points.push_back(candidate);
						cout << " 最末端的一个叶脉点的坐标" << candidate << endl;
					}
					else {
						cout << "候选最末端点与 vein1 最后一点相同，未加入。" << endl;
					}
				}

				//叶脉点云的存储和显示
				vein1->width = vein1->points.size();
				vein1->height = 1;
				vein1->is_dense = true;
				vein_size1 = vein1->size();
				cout << endl << "叶片片段数vein1->points.size()为" << vein1->points.size() << endl;
				std::stringstream  ss4;
				ss4 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-" << j << "-99" << ".pcd";
				writer.write<pcl::PointXYZ>("E:/develop/pcl_lri/data/20250521/cylinder - fuben2/" + ss4.str(), *vein1);
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color197(vein1, 0, 255, 0);	//显示各个圆柱叶片的重心点云，用绿色点标出
				viewer1->addPointCloud<pcl::PointXYZ>(vein1, single_color197, "d9d0");
				viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "d9d0");

				//叶长的计算和叶脉向量导入veinfang点云存储
				float leaf_len0 = 0;
				for (int i = 0; i < vein1->points.size() - 1; i++)
				{
					leaf_len0 = sqrt(pow((vein1->points[i + 1].x - vein1->points[i].x), 2) + pow((vein1->points[i + 1].y - vein1->points[i].y), 2)
						+ pow((vein1->points[i + 1].z - vein1->points[i].z), 2));
					leaf_len_array[leaf_len_i] = leaf_len0;
					leaf_len_i++;
					leaf_len = leaf_len + leaf_len0;

					fang.x = vein1->points[i + 1].x - vein1->points[i].x; // pcl::PointXYZ fang;//方向向量三个坐标值，最上面已经定义
					fang.y = vein1->points[i + 1].y - vein1->points[i].y;
					fang.z = vein1->points[i + 1].z - vein1->points[i].z;
					vec1.normalize();//单位化方法  Eigen::Vector4d vec1(fang.x, fang.y, fang.z, 0);//方向向量,最上面已经定义
					veinfang->points.push_back(fang);

				}//单独计算完叶长				
				cout << "总叶长" << leaf_len * 100 << "cm" << endl;


				const pcl::PointXYZ& last_vein2 = vein1->points.back(); //获取 vein1 的最后一个点
				float A[100], B[100], C[100], D[100];//分割平面定义出来A B C D 分割距离 H  
				int plane_to_vein_idx[100]; // 新增：记录每个平面对应的 vein1 索引（用于判断属于哪一部分）
				for (int ii = 0; ii < 100; ++ii) plane_to_vein_idx[ii] = -1;

				float H = Fb3;
				float H0 = Fb3;
				//float H00 = pcl::euclideanDistance(cloud_l_equ->points[flagz], vein1->points[vein_size1 - 1]);
				int j1 = 1;
				int vein_fang_size = vein_size1 - 1; //叶脉方向向量的数量要比叶脉点数量少一
				A[0] = veinfang->points[vein_fang_size - 1].x;//vein_fang_size
				B[0] = veinfang->points[vein_fang_size - 1].y;
				C[0] = veinfang->points[vein_fang_size - 1].z;
				//D[0] = -(A[0] * vein1->points[vein_size1 - 1].x + B[0] * vein1->points[vein_size1 - 1].y + C[0] * vein1->points[vein_size1 - 1].z);
				//D[0] = (H - H00) * sqrt(pow(A[0], 2) + pow(B[0], 2) + pow(C[0], 2)) - (A[0] * cloud_l_equ->points[flagz].x + B[0] * cloud_l_equ->points[flagz].y + C[0] * cloud_l_equ->points[flagz].z);
				D[0] = -(A[0] * last_vein2.x + B[0] * last_vein2.y + C[0] * last_vein2.z);
				plane_to_vein_idx[0] = vein_size1 - 1; // 记录对应 vein1 索引

				cout << "j1=0 " << "A0" << "  B  C  D " << A[0] << " " << B[0] << " " << C[0] << " " << D[0] << " " << endl;
				//开始最后一个叶脉点（最高点或最外点zmax）找分割平面，就是给数组A B C D 赋值
				for (int i = 0; i <= vein_size1 - 2; )
				{
					cout << "片段叶长leaf_len_array" << vein_fang_size - 1 - i << ": " << leaf_len_array[vein_fang_size - 1 - i] << " 分割距离H= " << H << endl;
					if (leaf_len_array[vein_fang_size - 1 - i] - H < 0)//如果小于0，这个点后有0个分割平面，从下一个点开始找
					{
						cout << "圆柱叶脉点序号[" << i << "]后面没有分割平面，直接更新H，到下一个叶脉点判断 " << endl;
						//变更一下距离H=H-leaf_len_array[vein_fang_size-1-i]，移到下一个点 i++ 开始找
						H = H - leaf_len_array[vein_fang_size - 1 - i];
						i++;
					}
					else//否则，大于0，这个点后面有1个或者2个分割平面
					{
						cout << "圆柱叶脉点序号[" << i << "]后面有分割平面： " << endl;
						if (leaf_len_array[vein_fang_size - 1 - i] - (H + H0) < 0)//如果小于0，有1个分割平面，找出存入数组，从下一个点开始找
						{
							A[j1] = veinfang->points[vein_fang_size - 1 - i].x;
							B[j1] = veinfang->points[vein_fang_size - 1 - i].y;
							C[j1] = veinfang->points[vein_fang_size - 1 - i].z;//方向向量的Z值应该都是正数
							//这个地方H前面应该是+，就是在往下面找分割平面方程了
							D[j1] = H * sqrt(pow(A[j1], 2) + pow(B[j1], 2) + pow(C[j1], 2)) - (A[j1] * vein1->points[vein_size1 - 1 - i].x + B[j1] * vein1->points[vein_size1 - 1 - i].y + C[j1] * vein1->points[vein_size1 - 1 - i].z);
							plane_to_vein_idx[j1] = vein_size1 - 1 - i; // 记录对应 vein1 索引
							cout << "j1=" << j1 << " A" << j1 << "  B  C  D " << A[j1] << " " << B[j1] << " " << C[j1] << " " << D[j1] << endl; //一个分割平面直接无空格输出
							j1++;
							//？不用管这种方法？对应maize-002ye1-(vein_size1-1-i-1).pcd  maize-002ye1-(vein_size1-1-i).pcd  maize-002ye1-(vein_size1-1-i+1).pcd  导进去重新分割出来了								
							H = H0 - (leaf_len_array[vein_fang_size - 1 - i] - H);
							i++;
						}
						else//否则，大于0，有2个分割平面，找出存入数组，从下一个点开始找
						{
							A[j1] = veinfang->points[vein_fang_size - 1 - i].x;
							B[j1] = veinfang->points[vein_fang_size - 1 - i].y;
							C[j1] = veinfang->points[vein_fang_size - 1 - i].z;
							D[j1] = H * sqrt(pow(A[j1], 2) + pow(B[j1], 2) + pow(C[j1], 2)) - (A[j1] * vein1->points[vein_size1 - 1 - i].x + B[j1] * vein1->points[vein_size1 - 1 - i].y + C[j1] * vein1->points[vein_size1 - 1 - i].z);
							plane_to_vein_idx[j1] = vein_size1 - 1 - i; // 记录对应 vein1 索引
							cout << "j1=" << j1 << "  A" << j1 << "  B  C  D " << A[j1] << " " << B[j1] << " " << C[j1] << " " << D[j1] << endl; //两个分割平面，第一个平面一个空格输出
							j1++;
							A[j1] = veinfang->points[vein_fang_size - 1 - i].x;
							B[j1] = veinfang->points[vein_fang_size - 1 - i].y;
							C[j1] = veinfang->points[vein_fang_size - 1 - i].z;
							D[j1] = (H + H0) * sqrt(pow(A[j1], 2) + pow(B[j1], 2) + pow(C[j1], 2)) - (A[j1] * vein1->points[vein_size1 - 1 - i].x + B[j1] * vein1->points[vein_size1 - 1 - i].y + C[j1] * vein1->points[vein_size1 - 1 - i].z);
							plane_to_vein_idx[j1] = vein_size1 - 1 - i;
							cout << "j1=" << j1 << "    A" << j1 << "  B  C  D " << A[j1] << " " << B[j1] << " " << C[j1] << " " << D[j1] << endl;  //两个分割平面，第二个平面两个空格输出
							j1++;
							H = H0 - (leaf_len_array[vein_fang_size - 1 - i] - (H0 + H)); //2H0 +H - leaf_len_array[vein_fang_size - 1 - i] 
							i++;
						}
					}

				}

				cout << endl;


				// 先找 vein1 中最高（z 最大）的叶脉点索引，用于判定分界
				int highest_idx = -1;
				float highest_z = -1e30f;
				if (!vein1->points.empty()) {
					for (size_t vi = 0; vi < vein1->points.size(); ++vi) {
						if (vein1->points[vi].z > highest_z) {
							highest_z = vein1->points[vi].z;
							highest_idx = (int)vi;
						}
					}
				}

				pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
				pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud2(new pcl::PointCloud<pcl::PointXYZ>());
				reader.read("E:/develop/pcl_lri/data/20250521/cylinder - fuben2/" + ss3.str() + "-222.pcd", *target_cloud2);
				pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud1(new pcl::PointCloud<pcl::PointXYZ>());
				reader.read("E:/develop/pcl_lri/data/20250521/cylinder - fuben2/" + ss3.str() + "-111.pcd", *target_cloud1);
				int bs_eq = 0; //等距分割的平面序号
				for (bs_eq = 0; bs_eq < j1; bs_eq++) // 等距平面分割循环，bs_eq为平面序号
				{
					*target_cloud = *cloud_l_equ; // 默认退化为全云						

					if (plane_to_vein_idx[bs_eq] >= 0 && highest_idx >= 0) {
						cout << "Plane " << bs_eq << " corresponds to vein index " << plane_to_vein_idx[bs_eq]
							<< ", highest vein index is " << highest_idx << endl;
						// 如果 plane 对应的 vein 索引 在最高点索引之后（你描述的“在最高处叶脉点后面计算出来的分割平面序号”）， 
						// 则该平面属于靠近茎秆的分割平面 -> 使用 cloud_part1（近茎）
						if (plane_to_vein_idx[bs_eq] <= highest_idx - 4) {

							*target_cloud = *target_cloud1;
						}
						else if (plane_to_vein_idx[bs_eq] <= highest_idx + 0) {
							*target_cloud = *cloud_l_equ;
						}
						else
							*target_cloud = *target_cloud2;
					}
					else {
						// 若没有 vein 索引信息或 highest_idx 无效，退化为使用原始 cloud_l_equ
						*target_cloud = *cloud_l_equ;
						cout << "Warning: plane_to_vein_idx or highest_idx invalid, using full cloud for segmentation." << endl;
					}

					equal_seg(A[bs_eq], B[bs_eq], C[bs_eq], D[bs_eq], bs_eq, target_cloud, ci_equ_1, j);
					// 每个截面曲线的显示ci_equ_1  可视化逻辑完全保留
					std::stringstream ss5;
					ss5 << time_id << "pianduan" << bs_eq;        //增加独一无二的ID，防止可视化覆盖
					pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(ci_equ_1, a[bs_eq % 14][0], a[bs_eq % 14][1], a[bs_eq % 14][2]);
					viewer2->addPointCloud<pcl::PointXYZ>(ci_equ_1, single_color, ss5.str());
				}
				//叶脉点云的存储和显示
				vein2->width = vein2->points.size();
				vein2->height = 1;
				vein2->is_dense = true;
				vein_size2 = vein2->size();
				cout << endl << "叶片片段数vein2->points.size()为" << vein2->points.size() << endl;
				std::stringstream  ss7;
				ss7 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-" << j << "-99" << ".pcd";
				writer.write<pcl::PointXYZ>("E:/develop/pcl_lri/data/20250521/equal_d - fuben2/" + ss7.str(), *vein2);
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color199(vein2, 0, 255, 0);	//显示各个圆柱叶片的重心点云，用绿色点标出
				viewer2->addPointCloud<pcl::PointXYZ>(vein2, single_color199, "d9d0dd");
				viewer2->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "d9d0dd");
				cout << endl;

				//等距叶宽计算
				cout << "每条曲线的欧氏距离叶宽如下列：" << endl;
				for (int bs_eq_i = 0; bs_eq_i <= bs_eq - 1; bs_eq_i++) {  //bs_eq在结束上个循环时会多+1，要取到所有截面曲线就要到bs_eq-1了
					//每个等距截面曲线依次导入ci_equ_2								
					std::stringstream  ss4;
					ss4 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-" << j << "-" << bs_eq_i << ".pcd";
					reader.read("E:/develop/pcl_lri/data/20250521/equal_d - fuben2/" + ss4.str(), *ci_equ_2);

					// 初始化最远距离和对应的点及点序号
					int fi1 = 0, fi2 = 0;
					float max_distance = 0.0;
					pcl::PointXYZ point11, point22;
					// 遍历所有点对，计算距离
					for (size_t i = 0; i < ci_equ_2->size(); ++i)
					{
						for (size_t j = i + 1; j < ci_equ_2->size(); ++j)
						{
							float distance = pcl::euclideanDistance(ci_equ_2->points[i], ci_equ_2->points[j]);
							if (distance > max_distance)
							{
								max_distance = distance;
								point11 = ci_equ_2->points[i];
								point22 = ci_equ_2->points[j];
								fi1 = i;
								fi2 = j;
							}
						}
					}
					//计算两点的直线距离，叶宽
					//cout<<bs_eq_i<<"截面曲线叶宽" << pcl::euclideanDistance(ci_equ_2->points[fi1], ci_equ_2->points[fi2]) << endl;					
					cout << sqrt(pow(ci_equ_2->points[fi1].x - ci_equ_2->points[fi2].x, 2) + pow(ci_equ_2->points[fi1].y - ci_equ_2->points[fi2].y, 2) + pow(ci_equ_2->points[fi1].z - ci_equ_2->points[fi2].z, 2)) << endl;
				}


			}//if导入叶片点云
		}//if一个茎秆对应叶片循环
	}//if导入茎秆点云

	while (!viewer1->wasStopped())
	{
		viewer1->spinOnce(1000);
	}
	while (!viewer2->wasStopped())
	{
		viewer2->spinOnce(1000);
	}
}


void cylinder_seg(pcl::PointXYZ center1,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_l_cly, pcl::PointCloud<pcl::PointXYZ>::Ptr ci_cly_1,
	int BS_cylinder, int* bs_cylinder, float k, bool k_is_valid, int j)
{
	// 在 xy 平面上得到叶片生长方向的单位向量 dir_xy
	Eigen::Vector2f dir_xy; //叶片生长方向的单位向量 dir_xy二维向量
	if (k_is_valid) {
		// 斜率 k 对应方向 (1, k)，将其单位化
		dir_xy = Eigen::Vector2f(1.0f, static_cast<float>(k));
		if (dir_xy.norm() > 1e-12f) dir_xy.normalize();
		else                        dir_xy = Eigen::Vector2f(1.0f, 0.0f);
	}
	else {
		// 竖直方向（dx≈0），在 xy 平面上方向为 (0, 1)
		dir_xy = Eigen::Vector2f(0.0f, 1.0f);
	}
	Eigen::Vector3f dir_xy_vert3d(-dir_xy.y(), dir_xy.x(), 0.0f);	// 垂直于生长方向的向量（在 xy 平面）

	pcl::PointIndices::Ptr inliers1(new pcl::PointIndices());  //点索引集合对象inliers---条件索引分割点云 

	const float z0 = center1.z;
	const float r = BS_cylinder * Fb1;// 分割半径
	Eigen::Vector3f P0(center1.x, center1.y, z0);
	//圆柱分割，以确定的方向为圆柱轴线方向，传输进来的center1为轴线上的一点
	for (size_t i = 0; i < cloud_l_cly->points.size(); i++) //按圆柱规则存储索引
	{
		const auto& pt = cloud_l_cly->points[i];
		Eigen::Vector3f P(pt.x, pt.y, pt.z);
		Eigen::Vector3f w = P - P0;
		float d2 = w.cross(dir_xy_vert3d).squaredNorm(); // 因为dir_xy_vert3d 已单位化，所以 d^2 = |w×a|^2
		if (1)
		{
			if (d2 < r * r) inliers1->indices.push_back((int)i);
		}

	}

	if (inliers1->indices.size() > 0) {   //按照索引进行分割并保存

		pcl::ExtractIndices<pcl::PointXYZ> extract0;
		extract0.setInputCloud(cloud_l_cly);
		extract0.setIndices(inliers1);
		extract0.setNegative(false);
		extract0.filter(*ci_cly_1);
		extract0.setNegative(true);
		extract0.filter(*cloud_l_cly);
		inliers1->indices.clear();

		//每个圆柱片段保存ci_cly_1								
		std::stringstream  ss4;
		ss4 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-" << j << "-" << *bs_cylinder << ".pcd";
		writer.write<pcl::PointXYZ>("E:/develop/pcl_lri/data/20250521/cylinder - fuben2/" + ss4.str(), *ci_cly_1);
		cout << "每个圆柱片段点数：" << ci_cly_1->points.size();//输出每个圆柱片段的点数

		//重心计算叶片叶脉点
		pcl::PointXYZ pm1;//先定义一个点
		double mx = 0, my = 0, mz = 0;
		for (size_t i = 0; i < ci_cly_1->points.size(); i++)
		{
			mx = mx + ci_cly_1->points[i].x;
			my = my + ci_cly_1->points[i].y;
			mz = mz + ci_cly_1->points[i].z;
		}
		pm1.x = mx / ci_cly_1->points.size();
		pm1.y = my / ci_cly_1->points.size();
		pm1.z = mz / ci_cly_1->points.size();
		//几何中心计算叶片叶脉点
		pcl::PointXYZ pm2;
		pcl::PointXYZ mind0, maxd0;
		pcl::getMinMax3D(*ci_cly_1, mind0, maxd0);
		pm2.x = (maxd0.x + mind0.x) / 2;
		pm2.y = (maxd0.y + mind0.y) / 2;
		pm2.z = (maxd0.z + mind0.z) / 2;
		cout << "  BS_cylinder = " << BS_cylinder << "  bs_cylinder = " << *bs_cylinder;	//输出的是圆柱序号和圆柱分割结果的片段点云的序号	
		if (pm == 1) {
			vein1->points.push_back(pm1);
			/*cout << " 重心叶脉点的坐标" << vein1->points[*bs_cylinder] << endl;  */
			cout << "  重心叶脉点的坐标" << pm1 << endl;
		}
		else {
			vein1->points.push_back(pm2);
			/*cout << "  几何中心叶脉点的坐标" << vein1->points[*bs_cylinder] << endl;  */
			cout << "  几何中心叶脉点的坐标" << pm2 << endl;
		}
		*bs_cylinder += 1;  //bs_cylinder+1,循环完之后输出bs_cylinder需要多加了1，也就是bs_cylinder其实正好是圆柱片段数量

	}

}

void equal_seg(float A, float B, float C, float D, int bs_eq,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_l_equ,
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_equ_1, int j)
{
	//创建存储内点的点索引对象inliers2  条件索引分割点云
	pcl::PointIndices::Ptr inliers2(new pcl::PointIndices());

	// 核心：等距平面分割的条件筛选，遍历点云筛选符合平面范围的点
	if (bs_eq == 0)
	{
		float norm = sqrt(A * A + B * B + C * C);
		// 若法向量近似为0，则退化为原有阈值判断（避免除零）
		if (norm < 1e-12f) {
			for (size_t n = 0; n < cloud_l_equ->points.size(); n++) {
				if (abs(A * cloud_l_equ->points[n].x + B * cloud_l_equ->points[n].y + C * cloud_l_equ->points[n].z + D)
					< Fb2 * sqrt(pow(A, 2) + pow(B, 2) + pow(C, 2)))
				{
					inliers2->indices.push_back(static_cast<int>(n));
				}
			}
		}
		else {
			int best_idx = 0;
			float best_val = 1e30f;
			for (size_t n = 0; n < cloud_l_equ->points.size(); ++n) {
				float val = fabs(A * cloud_l_equ->points[n].x + B * cloud_l_equ->points[n].y + C * cloud_l_equ->points[n].z + D) / norm; //好像是距离这第一个平面最近的点
				if (val < best_val) {
					best_val = val;
					best_idx = static_cast<int>(n);
				}
			}
			inliers2->indices.push_back(best_idx);
		}
	}
	else
	{
		for (size_t n = 0; n < cloud_l_equ->points.size(); n++)
		{
			// 你的核心判断公式：点到平面的距离 < 分辨率阈值  【公式完全保留，未做任何修改】
			if (abs(A * cloud_l_equ->points[n].x + B * cloud_l_equ->points[n].y + C * cloud_l_equ->points[n].z + D)
				< Fb2 * sqrt(pow(A, 2) + pow(B, 2) + pow(C, 2)))
			{
				inliers2->indices.push_back(n);
			}
		}
	}

	// 判断是否筛选出有效点云，有则执行分割+保存+可视化
	if (inliers2->indices.size() > 0)
	{
		// 创建滤波器对象 extract0，和圆柱分割用同一个类，逻辑完全一致
		pcl::ExtractIndices<pcl::PointXYZ> extract0;
		extract0.setInputCloud(cloud_l_equ);    // 载入待分割的原始点云
		extract0.setIndices(inliers2);          // 传入本次筛选的有效点索引
		extract0.setNegative(false);            // 设置：提取索引内的点(分割出目标层)
		extract0.filter(*ci_equ_1);             // 结果存入ci_equ_1：本次平面层的点云片段

		extract0.setNegative(true);             // 设置：提取索引外的点(剩余未分割点)
		extract0.filter(*cloud_l_equ);          // 核心：原地更新原云，抠除已分割的点，供下一轮循环使用
		inliers2->indices.clear();              // 清空索引，释放内存

		// 输出当前片段的信息（和你原代码一致的日志）
		cout << "等距曲线序号" << bs_eq << "片段";
		cout << "点数为" << ci_equ_1->points.size();

		//几何中心计算叶片叶脉点
		pcl::PointXYZ pm2;
		pcl::PointXYZ mind0, maxd0;
		pcl::getMinMax3D(*ci_equ_1, mind0, maxd0);
		pm2.x = (maxd0.x + mind0.x) / 2;
		pm2.y = (maxd0.y + mind0.y) / 2;
		pm2.z = (maxd0.z + mind0.z) / 2;
		cout << "  bs_eq = " << bs_eq;	//输出的是圆柱序号和圆柱分割结果的片段点云的序号	
		vein2->points.push_back(pm2);
		cout << "  几何中心叶脉点的坐标" << vein2->points[bs_eq] << endl;  //cout << "  几何中心叶脉点的坐标" << pm <<endl;


		// 每个截面曲线保存ci_equ_1  PCD文件保存逻辑完全保留，路径不变
		std::stringstream ss6;
		ss6 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-" << j << "-" << bs_eq << ".pcd";
		writer.write<pcl::PointXYZ>("E:/develop/pcl_lri/data/20250521/equal_d - fuben2/" + ss6.str(), *ci_equ_1);

	}
}