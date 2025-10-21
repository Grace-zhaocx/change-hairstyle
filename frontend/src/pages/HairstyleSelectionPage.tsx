import React, { useState, useEffect } from 'react';
import { Card, Button, Typography, Tabs, Form, Select, Slider, Input, Row, Col, message } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ScissorOutlined,
  EditOutlined,
  PictureOutlined,
  SettingOutlined,
  RocketOutlined
} from '@ant-design/icons';

const { Title, Paragraph } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;

interface HairstyleDescription {
  length: string;
  style: string;
  color: string;
  custom_description?: string;
}

interface HairstyleParameters {
  blend_strength: number;
  edge_smoothing: number;
  lighting_match: number;
}

const HairstyleSelectionPage: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [activeTab, setActiveTab] = useState('text');
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [imageId] = useState(location.state?.imageId);

  const [description, setDescription] = useState<HairstyleDescription>({
    length: 'medium',
    style: 'wavy',
    color: 'natural'
  });

  const [parameters, setParameters] = useState<HairstyleParameters>({
    blend_strength: 85,
    edge_smoothing: 70,
    lighting_match: 60
  });

  useEffect(() => {
    if (!imageId) {
      message.error('请先上传图片');
      navigate('/upload');
    }
  }, [imageId, navigate]);

  const handleTextDescriptionSubmit = async () => {
    if (!imageId) {
      message.error('图片信息缺失');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/v1/hairstyle/text-to-hair', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_id: imageId,
          description,
          parameters
        })
      });

      if (!response.ok) {
        throw new Error('提交失败');
      }

      const result = await response.json();
      message.success('任务创建成功！');

      // 跳转到处理页面
      navigate('/processing', {
        state: { taskId: result.data.task_id }
      });

    } catch (error) {
      message.error('提交失败，请重试');
    } finally {
      setLoading(false);
    }
  };

  const handleReferenceSubmit = async () => {
    // TODO: 实现参考图片换发型逻辑
    message.info('参考图片换发型功能开发中...');
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
            步骤 3/5: 选择您的理想发型
          </Title>
          <Paragraph className="text-gray-600">
            选择文本描述或参考图片方式来定义您想要的发型
          </Paragraph>
        </motion.div>

        <Row gutter={[24, 24]}>
          {/* Main Selection Area */}
          <Col xs={24} lg={16}>
            <motion.div variants={itemVariants}>
              <Card>
                <Tabs activeKey={activeTab} onChange={setActiveTab}>
                  <TabPane
                    tab={
                      <span>
                        <EditOutlined />
                        文本描述换发型
                      </span>
                    }
                    key="text"
                  >
                    <div className="p-4">
                      <Form form={form} layout="vertical">
                        <Form.Item label="发型长度">
                          <Select
                            value={description.length}
                            onChange={(value) => setDescription({...description, length: value})}
                            size="large"
                          >
                            <Select.Option value="short">短发</Select.Option>
                            <Select.Option value="medium">中长发</Select.Option>
                            <Select.Option value="long">长发</Select.Option>
                          </Select>
                        </Form.Item>

                        <Form.Item label="发型风格">
                          <Select
                            value={description.style}
                            onChange={(value) => setDescription({...description, style: value})}
                            size="large"
                          >
                            <Select.Option value="straight">直发</Select.Option>
                            <Select.Option value="wavy">微卷</Select.Option>
                            <Select.Option value="curly">大卷</Select.Option>
                            <Select.Option value="braided">编发</Select.Option>
                            <Select.Option value="buzzed">寸头</Select.Option>
                          </Select>
                        </Form.Item>

                        <Form.Item label="发色偏好">
                          <Select
                            value={description.color}
                            onChange={(value) => setDescription({...description, color: value})}
                            size="large"
                          >
                            <Select.Option value="natural">自然黑发</Select.Option>
                            <Select.Option value="brown">棕色系</Select.Option>
                            <Select.Option value="blonde">金色系</Select.Option>
                            <Select.Option value="black">纯黑色</Select.Option>
                            <Select.Option value="red">红色系</Select.Option>
                            <Select.Option value="custom">自定义</Select.Option>
                          </Select>
                        </Form.Item>

                        {description.color === 'custom' && (
                          <Form.Item label="自定义描述">
                            <TextArea
                              rows={3}
                              placeholder="请描述您想要的发色和发型细节..."
                              value={description.custom_description}
                              onChange={(e) => setDescription({
                                ...description,
                                custom_description: e.target.value
                              })}
                            />
                          </Form.Item>
                        )}

                        <Form.Item>
                          <Button
                            type="primary"
                            size="large"
                            icon={<RocketOutlined />}
                            onClick={handleTextDescriptionSubmit}
                            loading={loading}
                            className="btn-gradient w-full"
                          >
                            开始生成 →
                          </Button>
                        </Form.Item>
                      </Form>
                    </div>
                  </TabPane>

                  <TabPane
                    tab={
                      <span>
                        <PictureOutlined />
                        参考图片换发型
                      </span>
                    }
                    key="reference"
                  >
                    <div className="p-4">
                      <div className="text-center py-8">
                        <PictureOutlined style={{ fontSize: '64px', color: '#1890ff', marginBottom: '16px' }} />
                        <Title level={4}>上传参考发型图片</Title>
                        <Paragraph className="text-gray-500">
                          上传您喜欢的发型图片，AI将为您生成类似的发型
                        </Paragraph>

                        <div className="mt-6">
                          <Button
                            type="primary"
                            size="large"
                            icon={<PictureOutlined />}
                            onClick={handleReferenceSubmit}
                            className="btn-gradient"
                          >
                            选择参考图片
                          </Button>
                        </div>
                      </div>
                    </div>
                  </TabPane>
                </Tabs>
              </Card>
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
                </div>
              </Card>

              <motion.div variants={itemVariants}>
                <Card title="💡 预览效果">
                  <div className="text-center py-4">
                    <div className="bg-gray-100 rounded-lg p-4 mb-4">
                      <div className="bg-gray-200 h-32 rounded flex items-center justify-center">
                        <span className="text-gray-500">发型预览</span>
                      </div>
                    </div>
                    <Paragraph className="text-gray-500 text-sm">
                      基于您的选择，AI将生成相应的发型效果
                    </Paragraph>
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

export default HairstyleSelectionPage;