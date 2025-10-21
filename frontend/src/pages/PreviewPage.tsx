import React, { useState, useEffect } from 'react';
import { Card, Button, Typography, Row, Col, Slider, Space, Image, message } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  EyeOutlined,
  DownloadOutlined,
  ShareAltOutlined,
  SaveOutlined,
  SettingOutlined,
  ReloadOutlined
} from '@ant-design/icons';
import ReactCompareImage from 'react-compare-image';

const { Title, Paragraph } = Typography;

interface PreviewData {
  taskId: string;
  resultId: string;
  sourceImage: string;
  resultImage: string;
  parameters: any;
}

const PreviewPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [loading, setLoading] = useState(false);
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [parameters, setParameters] = useState({
    blend_strength: 85,
    edge_smoothing: 70,
    lighting_match: 60
  });

  const taskId = location.state?.taskId;

  useEffect(() => {
    if (!taskId) {
      message.error('缺少任务信息');
      navigate('/upload');
      return;
    }

    fetchPreviewData();
  }, [taskId, navigate]);

  const fetchPreviewData = async () => {
    try {
      setLoading(true);

      // 获取任务状态和结果
      const taskResponse = await fetch(`/api/v1/hairstyle/tasks/${taskId}`);
      if (!taskResponse.ok) {
        throw new Error('获取任务信息失败');
      }

      const taskData = await taskResponse.json();

      if (taskData.data.status !== 'completed' || !taskData.data.result_id) {
        message.error('任务尚未完成');
        navigate('/processing');
        return;
      }

      // 获取结果详情
      const resultResponse = await fetch(`/api/v1/hairstyle/results/${taskData.data.result_id}`);
      if (!resultResponse.ok) {
        throw new Error('获取结果信息失败');
      }

      const resultData = await resultResponse.json();

      setPreviewData({
        taskId: taskId,
        resultId: taskData.data.result_id,
        sourceImage: resultData.data.source_image_url,
        resultImage: resultData.data.result_image_url,
        parameters: resultData.data.result_params
      });

      // 设置参数滑块
      if (resultData.data.result_params) {
        setParameters({
          blend_strength: resultData.data.result_params.blend_strength || 85,
          edge_smoothing: resultData.data.result_params.edge_smoothing || 70,
          lighting_match: resultData.data.result_params.lighting_match || 60
        });
      }

    } catch (error) {
      console.error('Fetch preview data error:', error);
      message.error('加载预览失败');
    } finally {
      setLoading(false);
    }
  };

  const handleParameterChange = async () => {
    if (!previewData) return;

    try {
      setLoading(true);
      // TODO: 调用重新处理API
      message.info('参数调整功能开发中...');
    } catch (error) {
      message.error('参数调整失败');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!previewData) return;

    try {
      setLoading(true);

      // TODO: 调用保存API
      message.success('保存成功！');

      navigate('/result', {
        state: { resultId: previewData.resultId }
      });

    } catch (error) {
      message.error('保存失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!previewData) return;

    try {
      const response = await fetch(`/api/v1/hairstyle/results/${previewData.resultId}/download`);
      if (!response.ok) {
        throw new Error('获取下载链接失败');
      }

      const data = await response.json();

      // 创建下载链接
      const link = document.createElement('a');
      link.href = data.download_url;
      link.download = `hairstyle_result_${Date.now()}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      message.success('下载成功！');

    } catch (error) {
      message.error('下载失败');
    }
  };

  const handleShare = async () => {
    if (!previewData) return;

    try {
      // TODO: 调用分享API
      message.info('分享功能开发中...');
    } catch (error) {
      message.error('分享失败');
    }
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring',
        stiffness: 100
      }
    }
  };

  if (loading && !previewData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <Title level={4}>正在加载预览...</Title>
        </div>
      </div>
    );
  }

  if (!previewData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center">
        <Card>
          <Title level={4}>无法加载预览</Title>
          <Button type="primary" onClick={() => navigate('/upload')}>
            返回首页
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="container mx-auto px-4 py-8"
      >
        {/* Header */}
        <motion.div variants={itemVariants} className="text-center mb-8">
          <Title level={2} className="text-3xl font-bold mb-4">
            步骤 5/5: 预览并调整您的效果
          </Title>
          <Paragraph className="text-gray-600">
            您可以调整参数来优化效果，满意后保存您的换发型结果
          </Paragraph>
        </motion.div>

        <Row gutter={[24, 24]}>
          {/* Main Preview Area */}
          <Col xs={24} lg={16}>
            <motion.div variants={itemVariants}>
              <Card title="效果预览" className="mb-4">
                <div className="preview-container">
                  <ReactCompareImage
                    leftImage={previewData.sourceImage}
                    rightImage={previewData.resultImage}
                    leftImageLabel="原图"
                    rightImageLabel="换发型后"
                    sliderLineWidth={3}
                    handleSize={50}
                  />
                </div>

                <div className="text-center mt-4">
                  <Space size="middle">
                    <Button
                      icon={<EyeOutlined />}
                      onClick={() => {
                        // 全屏预览逻辑
                      }}
                    >
                      全屏预览
                    </Button>
                    <Button
                      icon={<ReloadOutlined />}
                      onClick={handleParameterChange}
                      loading={loading}
                    >
                      重新生成
                    </Button>
                  </Space>
                </div>
              </Card>

              {/* Action Buttons */}
              <motion.div variants={itemVariants}>
                <Card>
                  <div className="text-center">
                    <Title level={4} className="mb-4">您对这次换发型满意吗？</Title>

                    <Space size="large" className="mb-6">
                      <Button
                        type="primary"
                        size="large"
                        icon={<SaveOutlined />}
                        onClick={handleSave}
                        loading={loading}
                        className="btn-gradient"
                      >
                        保存结果
                      </Button>
                      <Button
                        size="large"
                        icon={<DownloadOutlined />}
                        onClick={handleDownload}
                      >
                        下载图片
                      </Button>
                      <Button
                        size="large"
                        icon={<ShareAltOutlined />}
                        onClick={handleShare}
                      >
                        分享给朋友
                      </Button>
                    </Space>

                    <div className="flex justify-center gap-2 mb-4">
                      {[1, 2, 3, 4, 5].map((star) => (
                        <Button
                          key={star}
                          type="text"
                          size="large"
                          icon={<span style={{ fontSize: '20px' }}>⭐</span>}
                          onClick={() => {
                            // 评分逻辑
                          }}
                        />
                      ))}
                    </div>

                    <Paragraph className="text-gray-500 text-sm">
                      点击星星为我们的AI评分
                    </Paragraph>
                  </div>
                </Card>
              </motion.div>
            </motion.div>
          </Col>

          {/* Parameters Sidebar */}
          <Col xs={24} lg={8}>
            <motion.div variants={itemVariants}>
              <Card title={<><SettingOutlined /> 效果调整</>} className="mb-4">
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between mb-2">
                      <span>发型融合度</span>
                      <span>{parameters.blend_strength}%</span>
                    </div>
                    <Slider
                      min={0}
                      max={100}
                      value={parameters.blend_strength}
                      onChange={(value) => setParameters({
                        ...parameters,
                        blend_strength: value
                      })}
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-2">
                      <span>边缘过渡</span>
                      <span>{parameters.edge_smoothing}%</span>
                    </div>
                    <Slider
                      min={0}
                      max={100}
                      value={parameters.edge_smoothing}
                      onChange={(value) => setParameters({
                        ...parameters,
                        edge_smoothing: value
                      })}
                    />
                  </div>

                  <div>
                    <div className="flex justify-between mb-2">
                      <span>光照匹配</span>
                      <span>{parameters.lighting_match}%</span>
                    </div>
                    <Slider
                      min={0}
                      max={100}
                      value={parameters.lighting_match}
                      onChange={(value) => setParameters({
                        ...parameters,
                        lighting_match: value
                      })}
                    />
                  </div>

                  <div className="pt-4">
                    <Button
                      type="primary"
                      block
                      onClick={handleParameterChange}
                      loading={loading}
                    >
                      应用调整
                    </Button>
                  </div>
                </div>
              </Card>

              <motion.div variants={itemVariants}>
                <Card title="💡 使用技巧">
                  <ul className="space-y-2 text-sm">
                    <li>• 拖动中间滑块查看对比效果</li>
                    <li>• 调整参数可以优化融合效果</li>
                    <li>• 点击全屏预览查看更多细节</li>
                    <li>• 保存后可以随时查看历史记录</li>
                  </ul>
                </Card>
              </motion.div>
            </motion.div>
          </Col>
        </Row>
      </motion.div>
    </div>
  );
};

export default PreviewPage;