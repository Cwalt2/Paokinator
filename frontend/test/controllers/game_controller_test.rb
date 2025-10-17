require "test_helper"

class GameControllerTest < ActionDispatch::IntegrationTest
  test "should get show" do
    get game_show_url
    assert_response :success
  end

  test "should get answer" do
    get game_answer_url
    assert_response :success
  end

  test "should get start" do
    get game_start_url
    assert_response :success
  end
end
