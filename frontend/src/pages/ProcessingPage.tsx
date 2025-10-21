import React, { useState, useEffect } from 'react';
import { Card, Progress, Typography, Button, Row, Col, Steps, Timeline } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  LoadingOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons';

const { Title, Paragraph } = Typography;
const { Step } = Steps;

interface ProcessingStatus {
  taskId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  currentStage: string;
  message?: string;
  estimatedTime?: number;
  error?: string;
}

const ProcessingPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [taskId] = useState(location.state?.taskId);
  const [status, setStatus] = useState<ProcessingStatus>({
    taskId: '',
    status: 'pending',
    progress: 0,
    currentStage: 'initializing'
  });

  const [isConnected, setIsConnected] = useState(false);
  const [ws, setWs] = useState<WebSocket | null>(null);

  const processingSteps = [
    {
      title: '人脸检测',
      description: '识别面部关键点',
      icon: <LoadingOutlined />
    },
    {
      title: '头发分割',
      description: '精确分离头发区域',
      icon: <LoadingOutlined />
    },
    {
      title: '发型生成',
      description: 'AI生成新发型',
      icon: <LoadingOutlined />
    },
    {
      title: '图像融合',
      description: '自然融合新发型',
      icon: <LoadingOutlined />
    },
    {
      title: '质量优化',
      description: '最终效果优化',
      icon: <LoadingOutlined />
    }
  ];

  useEffect(() => {
    if (!taskId) {
      navigate('/upload');
      return;
    }

    // 建立WebSocket连接
    const websocket = new WebSocket(`ws://localhost:8000/api/v1/ws/hairstyle/${taskId}`);

    websocket.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setStatus(prev => ({
          ...prev,
          ...data.data
        }));
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };

    websocket.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };

    websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    setWs(websocket);

    // 轮询备用方案
    const pollInterval = setInterval(async () => {
      if (!isConnected) {
        try {
          const response = await fetch(`/api/v1/hairstyle/tasks/${taskId}`);
          if (response.ok) {
            const result = await response.json();
            setStatus(result.data);
          }
        } catch (error) {
          console.error('Polling error:', error);
        }
      }
    }, 2000);

    return () => {
      websocket.close();
      clearInterval(pollInterval);
    };
  }, [taskId, navigate, isConnected]);

  useEffect(() => {
    if (status.status === 'completed') {
      setTimeout(() => {
        navigate('/preview', {
          state: { taskId: status.taskId }
        });
      }, 2000);
    }
  }, [status.status, navigate, status.taskId]);

  const getStepStatus = (stepTitle: string) => {
    const stageMap: { [key: string]: number } = {
      'initializing': 0,
      'face_detection': 1,
      'hair_segmentation': 2,
      'hairstyle_generation': 3,
      'image_blending': 4,
      'completed': 5
    };

    const currentStepIndex = stageMap[status.currentStage] || 0;

    if (stepTitle === '人脸检测' && currentStepIndex > 0) return 'finish';
    if (stepTitle === '头发分割' && currentStepIndex > 1) return 'finish';
    if (stepTitle === '发型生成' && currentStepIndex > 2) return 'finish';
    if (stepTitle === '图像融合' && currentStepIndex > 3) return 'finish';
    if (stepTitle === '质量优化' && currentStepIndex > 4) return 'finish';

    if (status.status === 'processing' && currentStepIndex > 0) {
      if (stepTitle === '人脸检测' && currentStepIndex === 1) return 'process';
      if (stepTitle === '头发分割' && currentStepIndex === 2) return 'process';
      if (stepTitle === '发型生成' && currentStepIndex === 3) return 'process';
      if (stepTitle === '图像融合' && currentStepIndex === 4) return 'process';
      if (stepTitle === '质量优化' && currentStepIndex === 5) return 'process';
    }

    return 'wait';
  };

  const getStepIcon = (stepTitle: string) => {
    const stepStatus = getStepStatus(stepTitle);
    if (stepStatus === 'finish') {
      return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
    }
    if (stepStatus === 'process') {
      return <LoadingOutlined style={{ color: '#1890ff' }} />;
    }
    return <ClockCircleOutlined style={{ color: '#d9d9d9' }} />;
  };

  const handleRetry = () => {
    navigate('/hairstyle-selection');
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

  if (status.status === 'failed') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-red-50 to-orange-50 flex items-center justify-center">
        <motion.div
          initial="hidden"
          animate="visible"
          variants={containerVariants}
          className="container mx-auto px-4 max-w-2xl"
        >
          <Card className="text-center">
            <ExclamationCircleOutlined style={{ fontSize: '64px', color: '#ff4d4f', marginBottom: '16px' }} />
            <Title level={3}>处理失败</Title>
            <Paragraph className="text-gray-600 mb-6">
              {status.error || '处理过程中遇到问题，请重试'}
            </Paragraph>
            <Button type="primary" icon={<ReloadOutlined />} onClick={handleRetry}>
              重新尝试
            </Button>
          </Card>
        </motion.div>
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
            步骤 4/5: AI正在为您生成新发型
          </Title>
          <Paragraph className="text-gray-600">
            请稍等片刻，我们的AI正在精心为您打造完美发型
          </Paragraph>
        </motion.div>

        <Row gutter={[24, 24]} justify="center">
          {/* Main Processing Card */}
          <Col xs={24} lg={16}>
            <motion.div variants={itemVariants}>
              <Card>
                <div className="text-center mb-8">
                  <div className="mb-6">
                    {status.status === 'processing' ? (
                      <LoadingOutlined style={{ fontSize: '64px', color: '#1890ff' }} />
                    ) : status.status === 'completed' ? (
                      <CheckCircleOutlined style={{ fontSize: '64px', color: '#52c41a' }} />
                    ) : (
                      <ClockCircleOutlined style={{ fontSize: '64px', color: '#faad14' }} />
                    )}
                  </div>

                  <Title level={3} className="mb-2">
                    {status.status === 'processing' && '正在处理中...'}
                    {status.status === 'completed' && '处理完成！'}
                    {status.status === 'pending' && '准备开始...'}
                  </Title>

                  <Paragraph className="text-lg text-gray-600 mb-6">
                    {status.message || '正在应用您选择的发型参数...'}
                  </Paragraph>

                  {/* Progress Bar */}
                  <div className="mb-6">
                    <Progress
                      type="circle"
                      percent={status.progress}
                      size={120}
                      strokeColor={{
                        '0%': '#108ee9',
                        '100%': '#87d068',
                      }}
                    />
                  </div>

                  {status.estimatedTime && (
                    <div className="text-gray-500">
                      预计剩余时间: {status.estimatedTime}秒
                    </div>
                  )}

                  {status.status === 'completed' && (
                    <div className="text-green-600">
                      正在跳转到预览页面...
                    </div>
                  )}
                </div>

                {/* Processing Steps */}
                <div className="mt-8">
                  <Title level={4} className="mb-4">处理步骤</Title>
                  <Steps direction="vertical" size="small">
                    {processingSteps.map((step, index) => (
                      <Step
                        key={index}
                        title={step.title}
                        description={step.description}
                        status={getStepStatus(step.title)}
                        icon={getStepIcon(step.title)}
                      />
                    ))}
                  </Steps>
                </div>
              </Card>
            </motion.div>
          </Col>

          {/* Info Sidebar */}
          <Col xs={24} lg={8}>
            <motion.div variants={itemVariants}>
              <Card title="💡 小知识" className="mb-4">
                <Timeline>
                  <Timeline.Item>AI会保持您的面部特征完全不变</Timeline.Item>
                  <Timeline.Item>新发型会根据您的脸型智能调整</Timeline.Item>
                  <Timeline.Item>处理时间通常为30-60秒</Timeline.Item>
                  <Timeline.Item>您可以在结果页面微调效果</Timeline.Item>
                </Timeline>
              </Card>

              <motion.div variants={itemVariants}>
                <Card title="🔧 技术信息">
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>任务ID:</span>
                      <span className="text-gray-500 text-sm">{taskId?.substring(0, 8)}...</span>
                    </div>
                    <div className="flex justify-between">
                      <span>连接状态:</span>
                      <span className={isConnected ? 'text-green-500' : 'text-orange-500'}>
                        {isConnected ? '已连接' : '连接中...'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>当前阶段:</span>
                      <span className="text-gray-500">{status.currentStage}</span>
                    </div>
                  </div>
                </Card>
              </motion.div>
            </motion.div>
          </Col>
        </Row>
      </motion.div>
    </div>
  );
};

export default ProcessingPage;