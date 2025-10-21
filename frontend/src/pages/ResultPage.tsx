import React, { useState, useEffect } from 'react';
import { Card, Button, Typography, Row, Col, Image, Rate, message, Modal } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  DownloadOutlined,
  ShareAltOutlined,
  HistoryOutlined,
  HeartOutlined,
  EyeOutlined,
  StarOutlined,
  RocketOutlined,
  CameraOutlined
} from '@ant-design/icons';

const { Title, Paragraph } = Typography;
const { confirm } = Modal;

interface ResultData {
  resultId: string;
  sourceImage: string;
  resultImage: string;
  qualityScore: number;
  downloadCount: number;
  viewCount: number;
  userRating?: number;
  userFeedback?: string;
  createdAt: string;
}

const ResultPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [loading, setLoading] = useState(false);
  const [resultData, setResultData] = useState<ResultData | null>(null);
  const [rating, setRating] = useState(0);
  const [feedback, setFeedback] = useState('');

  const resultId = location.state?.resultId;

  useEffect(() => {
    if (!resultId) {
      message.error('缺少结果信息');
      navigate('/upload');
      return;
    }

    fetchResultData();
  }, [resultId, navigate]);

  const fetchResultData = async () => {
    try {
      setLoading(true);

      const response = await fetch(`/api/v1/hairstyle/results/${resultId}`);
      if (!response.ok) {
        throw new Error('获取结果信息失败');
      }

      const data = await response.json();

      setResultData({
        resultId: data.data.result_id,
        sourceImage: data.data.source_image_url,
        resultImage: data.data.result_image_url,
        qualityScore: data.data.quality_score || 0,
        downloadCount: data.data.download_count,
        viewCount: data.data.view_count,
        userRating: data.data.user_rating,
        userFeedback: data.data.user_feedback,
        createdAt: data.data.created_at
      });

      if (data.data.user_rating) {
        setRating(data.data.user_rating);
      }
      if (data.data.user_feedback) {
        setFeedback(data.data.user_feedback);
      }

    } catch (error) {
      console.error('Fetch result data error:', error);
      message.error('加载结果失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    if (!resultData) return;

    try {
      const response = await fetch(`/api/v1/hairstyle/results/${resultData.resultId}/download`);
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

      // 更新下载计数
      setResultData(prev => prev ? {
        ...prev,
        downloadCount: prev.downloadCount + 1
      } : null);

      message.success('下载成功！');

    } catch (error) {
      message.error('下载失败');
    }
  };

  const handleShare = async () => {
    if (!resultData) return;

    try {
      // 生成分享链接
      const shareUrl = `${window.location.origin}/share/${resultData.resultId}`;

      // 复制到剪贴板
      if (navigator.clipboard) {
        await navigator.clipboard.writeText(shareUrl);
        message.success('分享链接已复制到剪贴板！');
      } else {
        // 降级方案
        const textArea = document.createElement('textarea');
        textArea.value = shareUrl;
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        message.success('分享链接已复制！');
      }

    } catch (error) {
      message.error('分享失败');
    }
  };

  const handleRating = async (value: number) => {
    if (!resultData) return;

    setRating(value);

    try {
      const response = await fetch(`/api/v1/hairstyle/results/${resultData.resultId}/feedback`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          rating: value,
          feedback: feedback
        })
      });

      if (!response.ok) {
        throw new Error('提交反馈失败');
      }

      message.success('感谢您的评价！');

    } catch (error) {
      message.error('评价提交失败');
    }
  };

  const handleFeedback = (value: string) => {
    setFeedback(value);
  };

  const handleNewExperience = () => {
    confirm({
      title: '开始新的体验',
      content: '这将创建一个新的换发型任务，当前结果会保存到历史记录中。',
      okText: '确定',
      cancelText: '取消',
      onOk() {
        navigate('/upload');
      }
    });
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

  if (loading && !resultData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto mb-4"></div>
          <Title level={4}>正在加载结果...</Title>
        </div>
      </div>
    );
  }

  if (!resultData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 flex items-center justify-center">
        <Card>
          <Title level={4}>无法加载结果</Title>
          <Button type="primary" onClick={() => navigate('/upload')}>
            返回首页
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      <motion.div
        initial="hidden"
        animate="visible"
        variants={containerVariants}
        className="container mx-auto px-4 py-8"
      >
        {/* Success Header */}
        <motion.div variants={itemVariants} className="text-center mb-8">
          <div className="mb-4">
            <span className="text-6xl">🎉</span>
          </div>
          <Title level={2} className="text-3xl font-bold mb-4 text-green-600">
            换发型成功完成！
          </Title>
          <Paragraph className="text-lg text-gray-600">
            您的新发型已经生成完成，快来看看效果吧！
          </Paragraph>
        </motion.div>

        <Row gutter={[24, 24]}>
          {/* Main Result Area */}
          <Col xs={24} lg={16}>
            <motion.div variants={itemVariants}>
              <Card title="您的换发型结果" className="mb-4">
                <Row gutter={[16, 16]}>
                  <Col xs={24} md={12}>
                    <div className="text-center">
                      <div className="mb-2">
                        <strong>原图</strong>
                      </div>
                      <Image
                        src={resultData.sourceImage}
                        alt="原图"
                        style={{ maxWidth: '100%', maxHeight: '300px' }}
                        className="rounded-lg shadow-md"
                      />
                    </div>
                  </Col>
                  <Col xs={24} md={12}>
                    <div className="text-center">
                      <div className="mb-2">
                        <strong>换发型后</strong>
                      </div>
                      <Image
                        src={resultData.resultImage}
                        alt="换发型后"
                        style={{ maxWidth: '100%', maxHeight: '300px' }}
                        className="rounded-lg shadow-md"
                      />
                    </div>
                  </Col>
                </Row>

                {/* Action Buttons */}
                <div className="text-center mt-6">
                  <Space size="middle">
                    <Button
                      type="primary"
                      size="large"
                      icon={<DownloadOutlined />}
                      onClick={handleDownload}
                      className="btn-gradient"
                    >
                      下载高清图
                    </Button>
                    <Button
                      size="large"
                      icon={<ShareAltOutlined />}
                      onClick={handleShare}
                    >
                      分享给朋友
                    </Button>
                    <Button
                      size="large"
                      icon={<EyeOutlined />}
                      onClick={() => {
                        // 全屏查看逻辑
                      }}
                    >
                      查看大图
                    </Button>
                  </Space>
                </div>
              </Card>

              {/* Feedback Section */}
              <motion.div variants={itemVariants}>
                <Card title="💬 您的反馈">
                  <div className="text-center">
                    <div className="mb-4">
                      <Title level={5}>请为这次体验评分</Title>
                      <Rate
                        value={rating}
                        onChange={handleRating}
                        style={{ fontSize: '24px' }}
                      />
                    </div>

                    <div className="mb-4">
                      <textarea
                        className="w-full p-3 border border-gray-300 rounded-lg"
                        rows={3}
                        placeholder="分享您的使用体验...（可选）"
                        value={feedback}
                        onChange={(e) => handleFeedback(e.target.value)}
                      />
                    </div>

                    {rating > 0 && (
                      <Button
                        type="primary"
                        onClick={() => handleRating(rating)}
                        className="btn-gradient"
                      >
                        提交反馈
                      </Button>
                    )}
                  </div>
                </Card>
              </motion.div>
            </motion.div>
          </Col>

          {/* Sidebar */}
          <Col xs={24} lg={8}>
            <motion.div variants={itemVariants}>
              <Card title="📊 结果统计" className="mb-4">
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span>📥 下载次数:</span>
                    <span className="font-semibold">{resultData.downloadCount}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>👁️ 查看次数:</span>
                    <span className="font-semibold">{resultData.viewCount}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>⭐ 质量评分:</span>
                    <span className="font-semibold">
                      {resultData.qualityScore ? `${(resultData.qualityScore * 100).toFixed(0)}%` : 'N/A'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>📅 生成时间:</span>
                    <span className="font-semibold text-sm">
                      {new Date(resultData.createdAt).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              </Card>

              <motion.div variants={itemVariants}>
                <Card title="🚀 推荐更多发型" className="mb-4">
                  <Row gutter={[8, 8]}>
                    {[1, 2, 3, 4].map((i) => (
                      <Col span={12} key={i}>
                        <div
                          className="bg-gray-100 rounded-lg p-4 text-center cursor-pointer hover:bg-gray-200 transition-colors"
                          onClick={() => {
                            navigate('/hairstyle-selection');
                          }}
                        >
                          <CameraOutlined style={{ fontSize: '24px', marginBottom: '8px' }} />
                          <div className="text-sm">发型 {i}</div>
                        </div>
                      </Col>
                    ))}
                  </Row>
                  <div className="text-center mt-4">
                    <Button type="link" onClick={() => navigate('/hairstyle-selection')}>
                      查看更多发型 →
                    </Button>
                  </div>
                </Card>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Card title="📜 历史记录">
                  <div className="text-center">
                    <Button
                      icon={<HistoryOutlined />}
                      onClick={() => {
                        // 跳转到历史记录页面
                        message.info('历史记录功能开发中...');
                      }}
                      block
                    >
                      查看历史记录
                    </Button>
                  </div>
                </Card>
              </motion.div>
            </motion.div>
          </Col>
        </Row>

        {/* CTA Section */}
        <motion.div variants={itemVariants} className="text-center mt-12">
          <Card>
            <Title level={3} className="mb-4">
              准备好尝试新的发型了吗？
            </Title>
            <Paragraph className="text-gray-600 mb-6">
              继续探索更多发型可能性，找到最适合您的完美造型
            </Paragraph>
            <Button
              type="primary"
              size="large"
              icon={<RocketOutlined />}
              onClick={handleNewExperience}
              className="btn-gradient text-lg px-8 py-4"
            >
              开始新的体验 →
            </Button>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default ResultPage;