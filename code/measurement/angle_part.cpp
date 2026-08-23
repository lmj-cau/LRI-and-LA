#include <vector>   
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


size_t time_id = 14; //第几次采集
size_t group_id = 5; //第几组
size_t plant_id = 6; //第几株
size_t leaf_id = 6; //第几片叶片
size_t leaf_maxid = 15;//最大的叶片序号

#define part_num 6  //等弧长的关键锚点个数（>=2），分的段数为part_num-1

//生成实现读写功能的对象
pcl::PCDWriter writer;
pcl::PCDReader reader;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("33D Viewer1"));
//定义圆柱分割的叶脉点云对象vein1(等距分割的为vein2)
pcl::PointCloud<pcl::PointXYZ>::Ptr vein1(new pcl::PointCloud<pcl::PointXYZ>());

// helper: 找到 cumulative length >= target 的第一个索引
static size_t find_index_by_target(const std::vector<float>& cum, float target) {
	for (size_t i = 0; i < cum.size(); ++i) {
		if (cum[i] >= target) return i;
	}
	return cum.size() - 1;
}

int main(int argc, char** argv)
{
	viewer1->setBackgroundColor(0, 0, 0);

	ofstream out1;
	out1.open("E:/develop/pcl_angle/data/20250521/phenotype - fuben/angle_part_5.csv", ios::app);
	out1.setf(ios::fixed);
	out1.setf(ios::showpoint);
	out1.precision(4);

	// 动态构建 CSV header：leaf_file,length_cm,angle_seg1_deg,...angle_segK_deg (K = part_num-1)
	if (out1.is_open()) {
		out1 << "leaf_file,length_cm";
		int segs = max(1, (int)part_num - 1);
		for (int s = 1; s <= segs; ++s) {
			out1 << ",angle_seg" << s << "_deg";
		}
		out1 << "\n";
	}
	else { cout << "ouput file error!!!!" << endl; }

	//定义一系列点云对象   
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_stem(new pcl::PointCloud<pcl::PointXYZ>());  // cloud_stem茎秆  
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a_point(new pcl::PointCloud<pcl::PointXYZ>());  // 茎秆点云上那一点  
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_leaf(new pcl::PointCloud<pcl::PointXYZ>());  //cloud_leaf叶片
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_cly_1(new pcl::PointCloud<pcl::PointXYZ>);//每个圆柱小片段点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_cly_2(new pcl::PointCloud<pcl::PointXYZ>);//
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_equ_1(new pcl::PointCloud<pcl::PointXYZ>);//每个等距分割片段点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr ci_equ_2(new pcl::PointCloud<pcl::PointXYZ>);//再次导入每隔一个等距截面曲线点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr veinfang(new pcl::PointCloud<pcl::PointXYZ>());	//叶脉方向向量点云veinfang，把叶脉点两两组成的方向向量按点云的方式存储起来

	//导入茎秆点云
	std::stringstream  ss0; //导入茎秆点云字符串
	ss0 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id;
	if (!reader.read("E:/develop/pcl_angle/data/20250521/origin/" + ss0.str() + "-0.pcd", *cloud_stem))
	{
		//显示茎秆点云
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color19(cloud_stem, 255, 255, 0);
		viewer1->addPointCloud<pcl::PointXYZ>(cloud_stem, single_color19, "dd");

		pcl::PointXYZ mind0, maxd0, center1;
		pcl::getMinMax3D(*cloud_stem, mind0, maxd0);
		center1.x = (maxd0.x + mind0.x) / 2;
		center1.y = (maxd0.y + mind0.y) / 2;

		//导入叶片的叶脉点云
		for (int j = leaf_id; j <= leaf_maxid; j++)
		{
			float leaf_len_array[100] = {}; //叶长数组
			int leaf_len_i = 0;       //小段叶片长序号     
			float leaf_len = 0;      //叶片总长初值
			int vein_size1 = 0; //圆柱分割片段叶脉点点云的个数
			int vein_size2 = 0; //等距截面曲线叶脉点点云的个数

			veinfang->clear();
			veinfang->width = 0;
			veinfang->height = 1;
			vein1->clear();          // 清空所有点数据
			vein1->width = 0;        // 重置宽度（可选，clear()已重置，但显式设置更严谨）
			vein1->height = 1;       // 重置为无序点云（根据你的场景调整，有序点云需对应设置）

			pcl::PointXYZ fang;//叶脉方向向量三个坐标值
			Eigen::Vector4d vec1(fang.x, fang.y, fang.z, 0);//定义的四维的叶脉方向向量，方便计算空间长度

			std::stringstream ss1;  //导入叶片叶脉点云的字符串
			ss1 << time_id << "/" << time_id << "-" << group_id << "-" << plant_id << "-" << j << "-99.pcd";
			if (!reader.read("E:/develop/pcl_angle/data/20250521/equal_d - fuben/" + ss1.str(), *vein1))
			{

				//显示叶脉点云
				pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color197(vein1, 0, 255, 0);	//显示各个圆柱叶片的重心点云，用绿色点标出
				viewer1->addPointCloud<pcl::PointXYZ>(vein1, single_color197, ss1.str());
				viewer1->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, ss1.str());

				//叶长的计算和叶脉向量导入veinfang点云存储
				float leaf_len0 = 0;
				for (int i = 0; i < vein1->points.size() - 1; i++) {
					leaf_len0 = pcl::euclideanDistance(vein1->points[i + 1], vein1->points[i]);
					leaf_len_array[leaf_len_i] = leaf_len0;
					leaf_len_i++;
					leaf_len = leaf_len + leaf_len0;

					fang.x = vein1->points[i + 1].x - vein1->points[i].x; // pcl::PointXYZ fang;//方向向量三个坐标值，最上面已经定义
					fang.y = vein1->points[i + 1].y - vein1->points[i].y;
					fang.z = vein1->points[i + 1].z - vein1->points[i].z;
					vec1.normalize();//单位化方法  Eigen::Vector4d vec1(fang.x, fang.y, fang.z, 0);//方向向量,最上面已经定义
					veinfang->points.push_back(fang);

				}
				cout << "叶片: " << ss1.str()
					<< "  长度: " << (leaf_len * 100.0) << " cm"
					<< endl;

				// --- 新增：通用分段取点并计算每段倾角（基部 -> ... -> 末端），锚点数量由 part_num 控制
				int M = max(2, part_num);            // 锚点数量（最少 2）
				int segs = M - 1;                    // 段数
				std::vector<float> angles(segs, 0.0f); // 每段角度占位

				size_t N = vein1->points.size();
				if (N >= 2 && (int)N >= M) {
					// 将原始点（末端->基部）逆序为基部->末端
					std::vector<pcl::PointXYZ> pts_bt; pts_bt.reserve(N);
					for (int k = static_cast<int>(N) - 1; k >= 0; --k) pts_bt.push_back(vein1->points[k]); // base -> tip

					// 计算累积弧长
					std::vector<float> cum(pts_bt.size(), 0.0f);
					for (size_t k = 1; k < pts_bt.size(); ++k) {
						cum[k] = cum[k - 1] + pcl::euclideanDistance(pts_bt[k], pts_bt[k - 1]);
					}
					float total_len = cum.back();
					// 目标位置按等弧长分布： i * (total_len / (M-1)), i=0..M-1
					std::vector<float> targets(M, 0.0f);
					for (int t = 0; t < M; ++t) targets[t] = (total_len * t) / (float)(M - 1);

					// 选择最近的点作为锚点（简洁实现）
					std::vector<pcl::PointXYZ> sel_pts(M);
					for (int t = 0; t < M; ++t) {
						size_t idx = find_index_by_target(cum, targets[t]);
						sel_pts[t] = pts_bt[idx];
					}

					// 计算每段向量并与全局竖直方向比较
					Eigen::Vector4f vz(0.0f, 0.0f, 1.0f, 0.0f);
					for (int s = 0; s < segs; ++s) {
						Eigen::Vector4f vseg(sel_pts[s + 1].x - sel_pts[s].x,
							sel_pts[s + 1].y - sel_pts[s].y,
							sel_pts[s + 1].z - sel_pts[s].z, 0.0f);
						float a = pcl::getAngle3D(vseg, vz, true); // 与竖直夹角（度）
						angles[s] = 90.0f - a; // 保持原代码习惯：转换为相对于水平面的角度
					}
				}
				else {
					// 点数不足时，回退到首尾线段角度并填充所有段
					Eigen::Vector4f v1(0.0f, 0.0f, 1.0f, 0);
					const pcl::PointXYZ& last_vein1 = vein1->points.back();
					Eigen::Vector4f v_ends(vein1->points[0].x - last_vein1.x, vein1->points[0].y - last_vein1.y, vein1->points[0].z - last_vein1.z, 0);
					float angle_ends = pcl::getAngle3D(v_ends, v1, true);
					angle_ends = 90 - angle_ends;
					for (int s = 0; s < segs; ++s) angles[s] = angle_ends;
				}

				// 输出：leaf_file,length_cm,angle_seg1,angle_seg2,...
				out1 << '"' << ss1.str() << '"' << "," << (leaf_len * 100.0);
				for (int s = 0; s < segs; ++s) out1 << "," << angles[s];
				out1 << "\n";

			}
		}
	}

	while (!viewer1->wasStopped())
	{
		viewer1->spinOnce(1000);
	}

}
